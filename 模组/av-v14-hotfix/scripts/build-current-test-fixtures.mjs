import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';

const root=process.argv[2];
if(!root)throw new Error('Usage: node scripts/build-current-test-fixtures.mjs /path/to/upstream');
const directory=new URL('../tests/fixtures/',import.meta.url);
const read=name=>JSON.parse(fs.readFileSync(new URL(name,directory),'utf8'));
const write=(name,value)=>fs.writeFileSync(new URL(name,directory),JSON.stringify(value,null,2)+'\n');
const sha=text=>createHash('sha256').update(text).digest('hex');
const capture=JSON.parse(fs.readFileSync(path.join(root,'../source-capture.json'),'utf8'));
function source(file){
  const text=fs.readFileSync(path.join(root,file),'utf8');
  if(sha(text)!==capture.files[file]?.sha256)throw new Error('Snapshot digest mismatch: '+file);
  return text;
}
const dsn=source('modules/dice-so-nice/main.js');
const provenance={module:'dice-so-nice',version:capture.versions['dice-so-nice'],file:'main.js',sha256:sha(dsn),capturedAt:capture.at,
  extraction:'Exact excerpts from the captured remote installation; bundle hashes are provenance only.'};
for(const name of ['dsn-chat-native.json','dsn-model-native.json','dsn-queue-native.json']){
  const fixture=read(name),methods=fixture.methods??fixture;
  const fragments={};
  for(const[key,text]of Object.entries(methods)){
    if(typeof text!=='string')continue;
    const start=dsn.indexOf(text);
    if(start<0)throw new Error('Changed DsN fragment: '+name+'/'+key);
    fragments[key]={start,end:start+text.length,sha256:sha(text)};
  }
  fixture.provenance={...provenance,fragments};
  write(name,fixture);
}
const quality=read('bbmm-dsn-native.json'),old=quality.dsn._prepareContext;
const start=dsn.indexOf(old.slice(0,150));
const end=dsn.indexOf('addTitleToOptions(',start);
if(start<0||end<0)throw new Error('DiceConfig context boundaries missing');
quality.dsn._prepareContext=dsn.slice(start,end);
const fragments={};
for(const[key,text]of Object.entries(quality.dsn)){
  const at=dsn.indexOf(text);if(at<0)throw new Error('Changed DiceConfig fragment: '+key);
  fragments[key]={start:at,end:at+text.length,sha256:sha(text)};
}
quality.provenance={...quality.provenance,at:capture.at,dsn:provenance.version,bbmm:capture.versions.bbmm,
  files:{...quality.provenance.files,dsn:sha(dsn)},dsnFragments:fragments};
write('bbmm-dsn-native.json',quality);

const settings=source('modules/bbmm/scripts/settings.js'),sync=source('modules/bbmm/scripts/setting-sync.js');
const registrations={};
for(const key of ['userSettingSync','enableUserSettingSync']){
  const start=settings.indexOf(`game.settings.register(BBMM_ID, "${key}", {`);
  const end=settings.indexOf('\n\t\t\t});',start)+'\n\t\t\t});'.length;
  if(start<0||end<start)throw new Error('Missing BBMM registration: '+key);
  registrations[key]=settings.slice(start,end);
}
const lockStart=sync.indexOf('async function _lc_writeLockChanges(');
const lockEnd=sync.indexOf('\n\t}',sync.indexOf('return { hardCount, softCount, removeCount };',lockStart))+4;
const writeLocks=sync.slice(lockStart,lockEnd);
write('bbmm-rules-native.json',{provenance:{module:'bbmm',version:capture.versions.bbmm,capturedAt:capture.at,
  files:{'scripts/settings.js':sha(settings),'scripts/setting-sync.js':sha(sync)},
  fragments:{...Object.fromEntries(Object.entries(registrations).map(([key,text])=>[key,{sha256:sha(text)}])),writeLocks:{sha256:sha(writeLocks)}}},registrations,writeLocks});

const turns=read('turn-lifecycle-native.json');
const consumers=[['reaction','modules/pf2e-reaction/pf2e-reaction.js','startTurn'],
  ['sustain','modules/pf2e-sustain-reminder/scripts/sustain-main.mjs','script'],
  ['summons','modules/pf2e-summons-assistant/scripts/specificClasses/necromancer.js','deleteItem']];
turns.currentProvenance={capturedAt:capture.at,consumers:{}};
for(const[key,file,fragment]of consumers){
  const text=source(file),method=turns[key][fragment],start=text.indexOf(method);
  if(start<0)throw new Error('Changed lifecycle callback: '+key);
  turns.currentProvenance.consumers[key]={file,version:capture.versions[file.split('/')[1]],sha256:sha(text),
    fragment:{start,end:start+method.length,sha256:sha(method)}};
}
write('turn-lifecycle-native.json',turns);
