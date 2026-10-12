import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,readdirSync} from 'node:fs';
import * as gates from '../scripts/world-gates.mjs';

const teams=['-','sog','pnvfcgjbf2cjp7gz','avecnu','team-automation-qa2'],fortress='ujx5r8oipw7ercdr';

test('ECNU shares every shield-chain gate with the other audited teams',()=>{
 for(const name of ['DESTRUCTIVE_BLOCK_WORLDS','RUNE_TRANSFER_WORLDS','NATIVE_IWR_STATUS_WORLDS'])assert.deepEqual([...gates[name]].sort(),[...teams].sort(),name);
 for(const name of ['SHIELD_BLOCK_WORLDS','SHIELD_WALL_COMPAT_WORLDS','REACTION_CHECKER_SETTING_WORLDS'])assert.deepEqual([...gates[name]].sort(),[...teams,fortress].sort(),name);
});

test('fortress and pilgrim features stay limited to their own worlds',()=>{
 for(const name of ['DEFENSIVE_ADVANCE_WORLDS','DESPERATE_PRAYER_WORLDS','ELECTRICITY_SETTLEMENT_WORLDS','FORCE_BARRAGE_WORLDS','FORTRESS_RULE_COMPAT_WORLDS','HALFLING_LUCK_WORLDS','ROARING_APPLAUSE_WORLDS','SPIRITUAL_SCAR_WORLDS'])assert.deepEqual([...gates[name]],[fortress],name);
 assert.deepEqual([...gates.GLIMPSE_WORLDS].sort(),[fortress,'team-automation-qa2'].sort());
 assert.deepEqual([...gates.SOG_PILGRIM_WORLDS].sort(),['sog','sog-pilgrim-qa'].sort());
});

test('world ids are written only in world-gates.mjs',()=>{
 const root=new URL('../scripts/',import.meta.url),ids=[...new Set(Object.values(gates).flatMap(gate=>[...gate]))].filter(id=>id!=='-');
 assert.ok(ids.includes('avecnu')&&ids.includes(fortress));
 for(const file of readdirSync(root,{recursive:true}).filter(file=>file.endsWith('.mjs')&&file!=='world-gates.mjs')){
  const source=readFileSync(new URL(file.replaceAll('\\','/'),root),'utf8');
  for(const id of ids)assert.equal(source.includes(`'${id}'`)||source.includes(`"${id}"`),false,`${file} hard-codes ${id}`);
 }
});

test('every exported gate is a set of world ids',()=>{
 for(const [name,gate] of Object.entries(gates)){assert.ok(gate instanceof Set,name);for(const id of gate)assert.equal(typeof id,'string',name);}
});
