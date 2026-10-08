import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
const root=process.argv[2];
if(!root)throw Error('Usage: node scripts/build-persistent-bridge-settings-test-fixture.mjs /path/to/upstream');
const sha=text=>createHash('sha256').update(text).digest('hex');
const capture=JSON.parse(fs.readFileSync(path.join(root,'../source-capture.json'),'utf8'));
const provenance={};
function read(file){
  const data=fs.readFileSync(path.join(root,'modules/pf2e-dsn-persistent-bridge/',file));
  const pin=capture.files['modules/pf2e-dsn-persistent-bridge/'+file];
  if(data.length!==pin?.bytes||sha(data)!==pin.sha256)throw Error('Capture differs: '+file);
  provenance[file]={sha256:sha(data),bytes:data.length};return data.toString('utf8');
}
const manifest=JSON.parse(read('module.json'));
if(manifest.id!=='pf2e-dsn-persistent-bridge'||manifest.version!=='0.5.4')throw Error('Unexpected bridge manifest');
const main=read('scripts/main.js'),settings=read('scripts/settings.js'),constants=read('scripts/constants.js');
function fragment(text,startText,endText){
  const start=text.indexOf(startText),end=text.indexOf(endText,start+startText.length);
  if(start<0||end<0)throw Error('Missing settings lifecycle fragment');
  return text.slice(start,end).trimEnd();
}
const createBridge=fragment(main,'export function createBridge(','export function installMessageSuppression(').slice(7);
const registerSettings=fragment(settings,'export function registerSettings(','export async function migrateSettings(').slice(7);
const declarations=fragment(constants,'export const MOD_ID=','export const getSetting=');
fs.writeFileSync(new URL('../tests/fixtures/persistent-bridge-settings-0.5.4-native.json',import.meta.url),
  JSON.stringify({createBridge,registerSettings,declarations,hashes:{createBridge:sha(createBridge),registerSettings:sha(registerSettings)},
    provenance:{module:manifest.id,version:manifest.version,capturedAt:capture.at,files:provenance}},null,2)+'\n');
console.log('Wrote the captured bridge enable/disable and setting registration fixture.');
