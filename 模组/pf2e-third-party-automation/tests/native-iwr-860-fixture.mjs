import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';

export const native860=JSON.parse(readFileSync(new URL('./fixtures/native-source-8.6.0.json',import.meta.url)));
export const hash=value=>createHash('sha256').update(value).digest('hex');
export function native860Source(){
 if(process.env.PF2E_NATIVE_860_BUNDLE){
  const source=readFileSync(process.env.PF2E_NATIVE_860_BUNDLE);
  assert.equal(hash(source),'d908c2fa1741d692b00b44e6dacba0bf7e8724d05898d555f9b640b2581d34b4');
  return source;
 }
 const r=native860.regions;
 // Only the audited seams are composed; this is not a complete system bundle.
 return Buffer.from(`class NativeActor {\n\t${r.nativeMethod}\n\tasync undoDamage() {}\n}\n${r.batchRegion}async function shiftAdjustDamage() {}\nclass FlatModifier {\n${r.flatRegion}\tasync afterRoll() {}\n}\n${r.stackingRegion}var StatisticModifier = null;\n${native860.init}\n`);
}
