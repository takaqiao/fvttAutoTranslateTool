import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {createHash} from 'node:crypto';

const root=process.argv[2];
if(!root)throw new Error('Usage: node scripts/build-dsn-643-test-fixture.mjs /path/to/upstream');
const directory=new URL('../tests/fixtures/',import.meta.url);
const read=name=>JSON.parse(fs.readFileSync(new URL(name,directory),'utf8'));
const sha=text=>createHash('sha256').update(text).digest('hex');
const capture=JSON.parse(fs.readFileSync(path.join(root,'../source-capture.json'),'utf8'));
const inventory=JSON.parse(fs.readFileSync(path.join(root,'../inventory.json'),'utf8'));
function source(file){
  const bytes=fs.readFileSync(path.join(root,file)),pin=capture.files[file];
  if(bytes.length!==pin?.bytes||sha(bytes)!==pin.sha256)throw new Error('Snapshot digest mismatch: '+file);
  return bytes.toString('utf8');
}
const bundle=source('modules/dice-so-nice/main.js');
const manifestText=source('modules/dice-so-nice/module.json'),manifest=JSON.parse(manifestText);
const row=inventory.rows['dice-so-nice'];
if(manifest.id!=='dice-so-nice'||manifest.version!=='6.4.3'||row?.version!==manifest.version
  ||row.sha256!==sha(manifestText))throw new Error('DsN manifest differs from inventory');

function classSource(name){
  const start=bundle.indexOf('class '+name+'{');
  if(start<0)throw new Error('Missing class: '+name);
  let depth=0,quote=null;
  for(let i=bundle.indexOf('{',start);i<bundle.length;i++){
    const char=bundle[i];
    if(quote){if(char==='\\')i++;else if(char===quote)quote=null;continue;}
    if(char==='"'||char==="'"||char==='`'){quote=char;continue;}
    if(char==='{')depth++;
    if(char==='}'&&--depth===0)return bundle.slice(start,i+1);
  }
  throw new Error('Unclosed class: '+name);
}
const fixture={queue:classSource('AnimationQueue'),accumulator:classSource('Accumulator'),
  boxClass:classSource('DiceBox'),engineClass:classSource('ThrowEngine')};
const context=vm.createContext({game:{settings:{get:()=>false}},DsnSettings:{isEnabled:()=>true}});
const classes=vm.runInContext(`(()=>{${fixture.accumulator};${fixture.queue};${fixture.boxClass};${fixture.engineClass};return {AnimationQueue,DiceBox,ThrowEngine};})()`,context);
const queue=new classes.AnimationQueue({});
const methods={boxStart:classes.DiceBox.prototype.startUnifiedBatch,engineStart:classes.ThrowEngine.prototype.startUnifiedBatch,
  boxAnimate:classes.DiceBox.prototype.animateThrow,engineComplete:classes.ThrowEngine.prototype.handlePersistentThrowCompletion,
  engineEffects:classes.ThrowEngine.prototype.handleSpecialEffectsInit,engineFinished:classes.ThrowEngine.prototype.throwFinished,
  engineResult:classes.ThrowEngine.prototype.fireResultEvents};
for(const[key,fn]of Object.entries(methods))fixture[key]=Function.prototype.toString.call(fn);
fixture.hashes=Object.fromEntries(Object.entries(fixture).map(([key,text])=>[key,sha(text)]));
fixture.hashes.onEnd=sha(Function.prototype.toString.call(queue.nextAnimation._onEnd));
fixture.hashes.attach=sha(Function.prototype.toString.call(classes.AnimationQueue.prototype.attach));
const legacy=read('dsn-queue-native.json');
fixture.completionConsumers={...legacy.completionConsumers};
for(const[key,text]of Object.entries(fixture.completionConsumers)){
  if(!fixture.boxAnimate.includes(text))throw new Error('Changed completion consumer: '+key);
  fixture.hashes[key]=sha(text);
}
for(const[key,expected]of Object.entries({queue:'6d28b4d7e4a9e6c998c04630536e705b1696213cc9e5c941d5eac36599f2f865',
  onEnd:'6ce7da61fbe7c5b249e4fd80011dbf30d87437e903b6b680ae4c4dca3f753b9a',
  boxAnimate:'c722cddba3608fade37e2b526ece24ce64d32ca144ce69bd39bebc072f5f3d12'})){
  if(fixture.hashes[key]!==expected)throw new Error('Unexpected current contract: '+key);
}
const fragments={};
function fragment(text){
  const start=bundle.indexOf(text);
  if(start<0)throw new Error('Fragment missing from captured bundle');
  return {start,end:start+text.length,sha256:sha(text)};
}
for(const[key,text]of Object.entries(fixture))if(typeof text==='string')fragments[key]=fragment(text);
const unchangedContracts={};
for(const name of ['dsn-chat-native.json','dsn-model-native.json','bbmm-dsn-native.json']){
  const prior=read(name),methods=prior.methods??prior.dsn;
  unchangedContracts[name]=Object.fromEntries(Object.entries(methods).map(([key,text])=>[key,fragment(text)]));
}
fixture.provenance={module:manifest.id,version:manifest.version,file:'main.js',sha256:sha(bundle),bytes:Buffer.byteLength(bundle),
  manifest:{file:'module.json',sha256:sha(manifestText),bytes:Buffer.byteLength(manifestText)},capturedAt:capture.at,
  extraction:'Exact excerpts from the captured installation; bundle hashes are provenance only.',fragments,unchangedContracts};
// Validate everything before writing; historical fixtures retain their original pins.
fs.writeFileSync(new URL('dsn-queue-6.4.3-native.json',directory),JSON.stringify(fixture,null,2)+'\n');
console.log('Wrote DsN '+manifest.version+' queue fixture; chat, model and quality contracts unchanged.');
