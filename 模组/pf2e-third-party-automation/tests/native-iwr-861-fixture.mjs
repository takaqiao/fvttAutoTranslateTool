import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';

export const native861=JSON.parse(readFileSync(new URL('./fixtures/native-source-8.6.1.json',import.meta.url)));
export const hash=value=>createHash('sha256').update(value).digest('hex');
export function native861Source(){
 if(process.env.PF2E_NATIVE_861_BUNDLE){
  const source=readFileSync(process.env.PF2E_NATIVE_861_BUNDLE);
  assert.equal(hash(source),'a48f919b8b9a791868ff3eb4eb560b210855932847e8c861ee914e16e3448ed3');
  return source;
 }
 const r=native861.regions;
 // Only the audited seams are composed; this is not a complete system bundle.
 return Buffer.from(`class NativeActor {\n\t${r.nativeMethod}\n\tasync undoDamage() {}\n}\n${r.batchRegion}async function shiftAdjustDamage() {}\nclass FlatModifier {\n${r.flatRegion}\tasync afterRoll() {}\n}\n${r.stackingRegion}var StatisticModifier = null;\n${native861.init}\n`);
}
