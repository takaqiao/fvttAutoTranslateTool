import test from 'node:test';
import assert from 'node:assert/strict';
import {createMetapowerProvider} from '../scripts/metapower/provider.mjs';
import {createMetapowerLedger,MODULE_ID as ID} from '../scripts/metapower/lifecycle.mjs';
import {METAPOWER_SOURCES} from '../scripts/metapower/rules.mjs';

function setup(t){
 const previous=globalThis.CONFIG;globalThis.CONFIG={Actor:{sheetClasses:{character:{}}},Dice:{rolls:[]}};
 t.after(()=>{if(previous===undefined)delete globalThis.CONFIG;else globalThis.CONFIG=previous});
 const owner={id:'owner'},stranger={id:'stranger'},gm={id:'gm'};
 const actor={uuid:'Actor.a',type:'character',flags:{[ID]:{metapower:{version:1,sequence:2,armed:{nonce:'original',kind:'widen',turn:null},receipts:{original:{status:'committed',nonce:'original'}}}}},items:new Map([['w',{sourceId:METAPOWER_SOURCES.widen}]]),
  testUserPermission:user=>user.id==='owner'||user.id==='gm',async update(changes){this.flags[ID].metapower=changes[`flags.${ID}.metapower`];return this}};
 const game={user:gm,users:{activeGM:gm},actors:new Map(),scenes:new Map(),modules:new Map(),pf2e:{actions:new Map()}};
 const ledger=createMetapowerLedger({game:{...game,user:gm},fromUuid:async uuid=>uuid===actor.uuid?actor:null});
 const hooks=new Map(),requests=[],errors=[];
 const provider=createMetapowerProvider({game,fromUuid:async uuid=>uuid===actor.uuid?actor:null,onError:e=>errors.push(e)});
 provider.register({Hooks:{on:(name,fn)=>hooks.set(name,fn)},libWrapper:{register(){}},socket:{register(){},async executeAsUser(method,_id,payload){
  requests.push({method,payload});try{return {ok:true,value:await ledger[method.replace('metapower:','')](payload,game.user)}}catch(error){return {ok:false,error:error.message}}
 }} });
 return {owner,stranger,gm,game,actor,hooks,requests,errors,provider};
}

test('current actor clear continues its original armed activation without native use or payment',async t=>{
 const f=setup(t);f.game.user=f.owner;
 assert.equal(typeof f.provider.clearArmed,'function');
 await f.provider.clearArmed(f.actor,{activationNonce:'original'});
 assert.equal(f.actor.flags[ID].metapower.armed,null);
 assert.deepEqual(f.actor.flags[ID].metapower.receipts,{original:{status:'committed',nonce:'original'}});
 assert.deepEqual(f.requests.map(r=>r.method),['metapower:clear']);
 assert.equal(f.requests[0].payload.activationNonce,'original');
});

test('stale actor clear cannot discard a newer activation and a foreign caller cannot clear either',async t=>{
 const f=setup(t);f.game.user=f.owner;
 assert.equal(typeof f.provider.clearArmed,'function');
 f.actor.flags[ID].metapower.armed={nonce:'new',kind:'siphoning',turn:null};
 await f.provider.clearArmed(f.actor,{activationNonce:'original'});
 assert.equal(f.actor.flags[ID].metapower.armed.nonce,'new');
 f.game.user=f.stranger;
 await assert.rejects(f.provider.clearArmed(f.actor,{activationNonce:'new'}),/权限|拥有/);
 assert.equal(f.actor.flags[ID].metapower.armed.nonce,'new');
});

test('GM rejects clearing an armed activation while another client has admitted a native action',async t=>{
 const f=setup(t),serverState=f.actor.flags[ID].metapower;
 const staleActor={...f.actor,flags:structuredClone(f.actor.flags)};serverState.pending='already-running';f.game.user=f.owner;
 await assert.rejects(f.provider.clearArmed(staleActor,{activationNonce:'original'}),/处理|执行/);
 assert.equal(f.actor.flags[ID].metapower.armed.nonce,'original');
 assert.equal(f.actor.flags[ID].metapower.pending,'already-running');
});

test('current actor action area exposes the owned activation clear with the captured nonce',async t=>{
 const f=setup(t);f.game.user=f.owner;
 const elements=[],root={ownerDocument:{createElement(){const element={listeners:{},remove(){elements.splice(elements.indexOf(this),1)},addEventListener(name,fn){this.listeners[name]=fn}};return element}},
  addEventListener(){},querySelectorAll(){return []},querySelector(selector){return elements.find(element=>selector==='.'+element.className)??null},append(element){elements.push(element)}};
 f.actor.isOwner=true;
 f.hooks.get('renderCharacterSheetPF2e')({actor:f.actor,isEditable:true},root);
 const button=elements.find(element=>element.className==='metapower-clear-armed');
 assert.ok(button,'armed activation must be accessible without its original chat card');
 f.actor.flags[ID].metapower.armed={nonce:'new',kind:'siphoning',turn:null};
 await button.listeners.click({preventDefault(){},stopPropagation(){}});
 assert.equal(f.requests.at(-1).payload.activationNonce,'original');
 assert.equal(f.actor.flags[ID].metapower.armed.nonce,'new');
 assert.deepEqual(f.errors,[]);
});
