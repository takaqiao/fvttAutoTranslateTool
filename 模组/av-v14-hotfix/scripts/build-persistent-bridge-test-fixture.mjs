import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';

const root=process.argv[2];
if(!root)throw Error('Usage: node scripts/build-persistent-bridge-test-fixture.mjs /path/to/upstream');
const sha=text=>createHash('sha256').update(text).digest('hex');
const capture=JSON.parse(fs.readFileSync(path.join(root,'../source-capture.json'),'utf8'));
const inventory=JSON.parse(fs.readFileSync(path.join(root,'../inventory.json'),'utf8'));
function read(file){
  const bytes=fs.readFileSync(path.join(root,file)),pin=capture.files[file];
  if(bytes.length!==pin?.bytes||sha(bytes)!==pin.sha256)throw Error('Snapshot digest mismatch: '+file);
  return bytes.toString('utf8');
}
const module='pf2e-dsn-persistent-bridge',file='modules/'+module+'/scripts/dsn-adapter.js';
const source=read(file),manifestText=read('modules/'+module+'/module.json'),manifest=JSON.parse(manifestText);
if(manifest.id!==module||manifest.version!=='0.5.4'||inventory.rows[module]?.version!==manifest.version
  ||inventory.rows[module].sha256!==sha(manifestText))throw Error('Bridge manifest differs from inventory');
if(sha(source)!=='82f6035a87e0f4269732a798ed88cb622ba8f036b3d40283bcfc041f9de9f0d8')throw Error('Unexpected bridge source');
const fixture={source,provenance:{module,version:manifest.version,file:'scripts/dsn-adapter.js',sha256:sha(source),
  bytes:Buffer.byteLength(source),capturedAt:capture.at,manifest:{sha256:sha(manifestText),bytes:Buffer.byteLength(manifestText)}}};
fs.writeFileSync(new URL('../tests/fixtures/persistent-bridge-0.5.4-native.json',import.meta.url),JSON.stringify(fixture,null,2)+'\n');
console.log('Wrote independent PersistentDice '+manifest.version+' adapter fixture.');
