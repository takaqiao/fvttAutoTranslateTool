import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {nativeRollEvent} from '../scripts/manual-native-roll.mjs';
const ID='pf2e-third-party-automation',enemy='Scene.s.Token.enemy';
function extracted(file,startMarker,endMarker,environment){const source=fs.readFileSync(new URL('../scripts/'+file,import.meta.url),'utf8'),start=source.indexOf(startMarker),end=source.indexOf(endMarker,start);assert(start>=0&&end>start);return vm.runInNewContext('('+source.slice(start,end).trim()+')',environment)}
for(const setting of [false,true])test(`campaign Strike and damage each preserve native acceptance with preference ${setting}`,async()=>{
 const events=[],game={user:{settings:{showCheckDialogs:setting,showDamageDialogs:setting}}},target={uuid:enemy,actor:{uuid:'Actor.enemy',hasCondition:()=>false},object:{}},weapon={uuid:'Actor.pc.Item.weapon',isMelee:true};
 const strike={ready:true,item:weapon,variants:[{async roll(options){events.push(options.event);await options.callback({options:{degreeOfSuccess:2}},'success',{id:'check',flags:{pf2e:{context:{}}}});return {total:20}}}],damage:async options=>{events.push(options.event);return {total:8}}};
 const run=extracted('campaign-feats.mjs','async function compositeStrike(','async function linkDamageCard(',{game,MODULE_ID:ID,oneTarget:async()=>target,values:c=>[...c],pick:async()=> '0:0',requireMeleeTarget(){},owner(){},nextOwnTurnTiming:()=>null,nativeRollEvent,saveFlag:async()=>{},opts:()=>[]});await run({actor:{uuid:'Actor.pc',system:{actions:[strike]}},item:{},message:{id:'use'},user:{},action:'campaign:slam'});assert.equal(events.length,2);for(const event of events)assert.equal(event.shiftKey,!setting);
});
test('campaign Battle Medicine cancellation preserves its real native skill check window',async()=>{
 const target={actor:{items:[]}},stat={rank:1,check:{async roll(options){assert.equal(options.skipDialog,false);assert.equal(options.event,null);return null}}};
 const run=extracted('campaign-feats.mjs','async function battleMedicine(','async function processCheck(',{MODULE_ID:ID,oneTarget:async()=>target,distance:()=>5,values:x=>[...x],hasSource:()=>false,own:()=>({}),BM_IMMUNITY:'immunity',pick:async()=> '1'});await assert.rejects(run({actor:{uuid:'Actor.pc',items:[],skills:{medicine:stat}},item:{},message:{id:'use'},user:{}}),/取消/);
});
