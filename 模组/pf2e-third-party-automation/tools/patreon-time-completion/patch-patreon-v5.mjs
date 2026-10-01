import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';

export const BASE_SHA='89ded325b92fa6b03dcf9257337ae2d628b3e99c987fe22cef9fff4e1837f4e9';
export const MANIFEST_SHA='1751242ef5858ee55bb33a79194ddbce5b2109f35e89d4d64df0bf4cb6fc012a';
const directory=path.dirname(fileURLToPath(import.meta.url));
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
function once(source,before,after){if(source.split(before).length!==2)throw Error('patch-anchor-not-unique');return source.replace(before,after);}
export function patchPatreon(bytes){
 if(hash(bytes)!==BASE_SHA)throw Error('patreon-source-sha-mismatch');
 let source=Buffer.from(bytes).toString('utf8');
 const start=source.indexOf('Hooks.on("updateWorldTime",'),end=source.indexOf('r(Mn,"handleFastHealingTime");',start)+'r(Mn,"handleFastHealingTime");'.length;
 if(start<0||end<=start)throw Error('patch-region-unavailable');
 const original=source.slice(start,end);let patched=original;
 patched=once(patched,'Hooks.on("updateWorldTime",(a,e,t,i)=>{k()&&(y("fastHealingTime")&&Mn(e),Rn())});',
  'Hooks.on("updateWorldTime",(a,e,t,i)=>{if(k()&&__patreonTimeCompletion.owned()){const x=__patreonTimeCompletion.begin(a,e,t,i),work=[];if(!__patreonTimeCompletion.authorize(x))return;try{y("fastHealingTime")&&work.push(Mn(e,x));work.push(Rn(x));}catch(error){work.push(Promise.reject(error));__patreonTimeCompletion.publish(x,work);throw error;}__patreonTimeCompletion.publish(x,work);}});');
 patched=once(patched,'function Rn(){game.actors.map','function Rn(x){return __patreonTimeCompletion.branches(game.actors.map');
 patched=once(patched,'.forEach(async e=>{let t=e.getFlag(u,"decrementPeriod");','.map(async e=>{let t=e.getFlag(u,"decrementPeriod");');
 patched=once(patched,'await e.update({"system.start.value":game.time.worldTime}),e.decrease())})}',
  'await __patreonTimeCompletion.update(x,e,{"system.start.value":game.time.worldTime},"effect-start"),__patreonTimeCompletion.decrease(x,e))}),x)}');
 patched=once(patched,'async function Mn(a){','async function Mn(a,x){');
 patched=once(patched,'.filter(l=>l.test())','.filter(l=>__patreonTimeCompletion.predicate(x,l,l.test()))');
 patched=once(patched,'let c=l.resolveValue(l.value);','let c=__patreonTimeCompletion.resolved(x,l,l.resolveValue(l.value),"eligibility");');
 patched=once(patched,'let s=0,l=game.actors.get(o);if(','let s=0,l=game.actors.get(o);__patreonTimeCompletion.member(x,o,l);if(');
 patched=once(patched,'let m=await new h.DamageRoll(`{(${f.resolveValue(f.value)})[healing]}`).evaluate();',
  'let m=__patreonTimeCompletion.roll(x,f,await new h.DamageRoll(`{(${__patreonTimeCompletion.resolved(x,f,f.resolveValue(f.value),"roll")})[healing]}`).evaluate());');
 patched=once(patched,'l.update({"system.attributes.hp.value":c}):l.update({"system.attributes.hp.value":l.system.attributes.hp.max})',
  '__patreonTimeCompletion.update(x,l,{"system.attributes.hp.value":c},"actor-hp"):__patreonTimeCompletion.update(x,l,{"system.attributes.hp.value":l.system.attributes.hp.max},"actor-hp")');
 const observer=fs.readFileSync(path.join(directory,'completion-observer-v5.js'),'utf8');
 return {bytes:Buffer.from(source.slice(0,start)+observer+patched+source.slice(end)),originalRegion:original,patchedRegion:observer+patched,start,end};
}
export function preparePatreon(sourcePath,manifestPath,outputDirectory){
 const output=path.resolve(outputDirectory),allowed=path.resolve(directory);
 if(path.dirname(output)!==allowed||fs.realpathSync(directory)!==allowed)throw Error('patch-output-outside-seam-directory');
 if(fs.existsSync(output))throw Error('patch-output-already-exists');
 const original=fs.readFileSync(sourcePath),manifest=fs.readFileSync(manifestPath);
 if(hash(manifest)!==MANIFEST_SHA)throw Error('patreon-manifest-sha-mismatch');
 if(JSON.parse(manifest.toString('utf8')).version!=='3.2.29')throw Error('patreon-version-mismatch');
 const result=patchPatreon(original);fs.mkdirSync(output,{recursive:false});
 fs.writeFileSync(path.join(output,'original-index.js'),original,{flag:'wx'});fs.writeFileSync(path.join(output,'original-module.json'),manifest,{flag:'wx'});
 fs.writeFileSync(path.join(output,'patched-index.js'),result.bytes,{flag:'wx'});
 const inventory={protocol:'patreon-native-time-seam.v1',status:'prepared-not-installed',source:{file:path.resolve(sourcePath),sha256:hash(original)},manifest:{file:path.resolve(manifestPath),sha256:hash(manifest)},
  files:['original-index.js','original-module.json','patched-index.js'].map(file=>({file,sha256:hash(fs.readFileSync(path.join(output,file)))})),region:{start:result.start,end:result.end,originalSHA256:hash(result.originalRegion),patchedSHA256:hash(result.patchedRegion)},patcherSHA256:hash(fs.readFileSync(fileURLToPath(import.meta.url))),observerSHA256:hash(fs.readFileSync(path.join(directory,'completion-observer-v5.js')))};
 fs.writeFileSync(path.join(output,'manifest.json'),JSON.stringify(inventory,null,2)+'\n',{flag:'wx'});
 fs.writeFileSync(path.join(output,'time-handler.diff'),`--- original-handler\n+++ patched-handler\n@@ original characters ${result.start}-${result.end} @@\n-${result.originalRegion}\n+${result.patchedRegion.split('\n').join('\n+')}\n`,{flag:'wx'});return inventory;
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const [sourcePath,manifestPath,outputDirectory]=process.argv.slice(2);if(!sourcePath||!manifestPath||!outputDirectory)throw Error('usage: patch-patreon source manifest new-output-directory');
 console.log(JSON.stringify(preparePatreon(sourcePath,manifestPath,outputDirectory)));
}
