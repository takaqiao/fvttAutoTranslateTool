import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
const roots=process.argv.slice(2);
if(roots.length!==2)throw Error('Usage: node scripts/build-dsn-worker-test-fixture.mjs /current/upstream /legacy/upstream');
const sha=text=>createHash('sha256').update(text).digest('hex');
const exec='exec(i,r=null,o=[],c){return new Promise((d,p)=>{const g=this._messageId++;this._messages.set(g,[d,p,c]),this._worker.postMessage([g,r,i],o||[])})}';
const pins={'6.4.3':'c5d68e23c907f11e63c008066c639dce7a6f32c3f14edb90088364d4fe3dbdbe',
  '6.4.2':'8ea57ede46b6b0f6e4367c1d9b1574c93ead3a95e981810945b53e7daf18e236'};
const provenance=[];
for(const root of roots){
  const capture=JSON.parse(fs.readFileSync(path.join(root,'../source-capture.json'),'utf8'));
  function read(file){
    const data=fs.readFileSync(path.join(root,file)),pin=capture.files[file];
    if(data.length!==pin?.bytes||sha(data)!==pin.sha256)throw Error('Capture differs: '+file);
    return data.toString('utf8');
  }
  const source=read('modules/dice-so-nice/main.js'),manifest=JSON.parse(read('modules/dice-so-nice/module.json'));
  const start=source.indexOf(exec);
  if(manifest.id!=='dice-so-nice'||sha(source)!==pins[manifest.version]||start<0||source.indexOf(exec,start+1)>=0)
    throw Error('Unsupported worker source');
  provenance.push({version:manifest.version,sha256:sha(source),bytes:Buffer.byteLength(source),capturedAt:capture.at,
    fragment:{start,end:start+exec.length}});
}
if(new Set(provenance.map(pin=>pin.version)).size!==2)throw Error('Both source profiles are required');
fs.writeFileSync(new URL('../tests/fixtures/dsn-worker-native.json',import.meta.url),JSON.stringify({exec,sha256:sha(exec),provenance},null,2)+'\n');
console.log('Wrote independently verified DsN worker RPC fixture for both profiles.');
