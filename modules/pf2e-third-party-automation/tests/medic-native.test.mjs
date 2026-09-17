import test from 'node:test';
import assert from 'node:assert/strict';
import {existsSync,readFileSync} from 'node:fs';
import {createMedicNative,pinnedMedicTarget} from '../scripts/medic-native.mjs';
const M='pf2e-third-party-automation';
test('pinned actor scope reads native private fields and overrides locked selections without changing the actor',()=>{
 class Patient{#id='patient';get uuid(){return `Actor.${this.#id}`;}readId(){return this.#id;}}
 const actor=new Patient(),wrong={id:'wrong'},target={id:'target',actor};
 Object.defineProperty(actor,'getActiveTokens',{value:()=>[wrong],enumerable:true});
 const pinned=pinnedMedicTarget(target);
 assert.ok(pinned instanceof Patient);assert.equal(pinned.uuid,'Actor.patient');assert.equal(pinned.readId(),'patient');
 assert.deepEqual(pinned.getActiveTokens(true,true),[target]);assert.deepEqual(actor.getActiveTokens(),[wrong]);
 assert.equal('getActiveTokens'in pinned,true);assert.ok(Object.keys(pinned).includes('getActiveTokens'));
 assert.deepEqual(Object.getOwnPropertyDescriptor(pinned,'getActiveTokens').value(),[target]);
});
test('native Poison and First Aid receive exact actor/target, continuation and player variant',async()=>{
 const calls=[],actor={uuid:'Actor.healer'},patient={uuid:'Actor.patient'},target={uuid:'Scene.s.Token.t',actor:patient},healer={object:{id:'h'}},user={id:'owner'},continuation={actorUuid:actor.uuid,cardId:'card',nonce:'nonce'};
 const game={pf2e:{actions:new Map(['treat-poison','administer-first-aid'].map(key=>[key,{use:async args=>{calls.push([key,args]);return [{actor}];}}]))}};
 const delegate=createMedicNative({game,choose:async()=> 'stabilize'});
 for(const branch of ['treat-poison','administer-first-aid'])await delegate({actor,target,healer,user,branch,continuation,validate:()=>{}});
 assert.equal(calls.length,2);for(const [,args]of calls){assert.deepEqual(args.actors,[actor]);assert.equal(args.target,patient);assert.deepEqual(args[M].metapowerContinuation,continuation);}
 assert.equal(calls[1][1].variant,'stabilize');
});
function detachedWorkbench({cancel=false,macroSource=null}={}){
 let release;const gate=new Promise(resolve=>release=resolve),hooks=new Map(),cards=[],observed={};
 const Hooks={on(name,fn){hooks.set(fn,name);return fn;},off(_name,fn){hooks.delete(fn);},once(name,fn){const wrapped=(...args)=>{hooks.delete(wrapped);return fn(...args);};hooks.set(wrapped,name);return wrapped;},call(name,...args){for(const [fn,key]of [...hooks])if(key===name)fn(...args);}};
 const actor={id:'healer',uuid:'Actor.healer',skills:{}},patient={id:'patient',uuid:'Actor.patient'},healer={id:'h',uuid:'Scene.s.Token.h'},target={id:'t',uuid:'Scene.s.Token.t',actor:patient};healer.object={id:'h',actor};target.object={id:'t',actor:patient};
 const game={user:{id:'gm',targets:new Set([{id:'wrong'}]),getFlag:()=>true},system:{id:'pf2e'},settings:{get:()=>false},combats:{active:null},messages:new Map(),modules:new Map([['xdy-pf2e-workbench',{active:true}]]),packs:new Map()};
 // Foundry defines canvas.tokens as an immutable own layer property; a scope must not proxy over it directly.
 const canvas=Object.defineProperty({},'tokens',{value:{controlled:[{id:'wrong'}]},enumerable:true});
 actor.skills.medicine={async roll(args){observed.check=args;const form={addEventListener(_event,handler){observed.submit=handler;},removeEventListener(){}};Hooks.call('renderCheckModifiersDialog',{context:{options:new Set(args.extraRollOptions)}},[form]);await gate;if(cancel)return null;const roll={total:25,options:{degreeOfSuccess:2}},message={id:'native-check',async update(changes){this.flags[M]={medicWorkbench:changes['flags.'+M+'.medicWorkbench']};},speaker:{actor:actor.id},rolls:[roll],flags:{pf2e:{context:{type:'skill-check',options:args.extraRollOptions,target:{actor:patient.uuid,token:target.uuid}}}}};game.messages.set(message.id,message);await args.callback(roll,'success',message);return roll;}};
 class ChatMessage{static #speaker={actor:actor.id};static getSpeaker(){return this.#speaker;}static async create(data){const m={...data,id:`card${cards.length}`};cards.push(m);return m;}}
 class DamageRoll{_total=8;async roll(){return this;}async toMessage(data){return ChatMessage.create(data);}}
 class CheckRoll{total=22;async roll(){return this;}}
 class Dialog{constructor(options){this.options=options;observed.dialog=this;}render(){return this;}}
 const macro={async execute(scope){observed.scope=scope;new scope.Dialog({buttons:{yes:{async callback(){// Workbench's actual outer callback intentionally does not await rollTreatWounds.
  void scope.token.actor.skills.medicine.roll({dc:{value:20},extraRollOptions:['action:treat-wounds'],callback:async()=>{observed.healReady=()=>new (scope.CONFIG??{Dice:{rolls:[DamageRoll]}}).Dice.rolls[0]().toMessage({speaker:scope.ChatMessage.getSpeaker(),flags:{treat_wounds_battle_medicine:{id:target.id,healerId:actor.id,dos:2,healing:8}}});}});
 }},no:{}},render(){}});}};
 if(macroSource){
  actor.items=[{type:'feat',slug:'battle-medicine'}];actor.itemTypes={feat:actor.items,effect:[],equipment:[{slug:'healers-toolkit',handsHeld:0}]};actor.system={details:{level:{value:5}}};actor.getRollOptions=()=>['self:type:character','self:trait:elf','self:effect:charged','feat:battle-medicine'];Object.assign(actor.skills.medicine,{rank:1,label:'medicine',modifiers:[{type:'proficiency',modifier:7}]});patient.items=[];patient.itemTypes={effect:[]};
  game.modules.set('dice-so-nice',{active:true});game.packs.set('xdy-pf2e-workbench.asymonous-benefactor-macros',{index:[]});
  const AsyncFunction=Object.getPrototypeOf(async function(){}).constructor;
  macro.execute=async scope=>{observed.scope=scope;const globals={...scope,Hooks,event:null,fromUuid:async()=>({toObject:()=>({name:'Immunity',system:{tokenIcon:{},duration:{value:1,unit:'days'}},flags:{}})}),ui:{notifications:{warn:message=>{throw Error(message);},info(){}}},console:{log(){}},CONST:{CHAT_MESSAGE_STYLES:{ROLL:5,OTHER:0}}};return new AsyncFunction(...Object.keys(globals),macroSource)(...Object.values(globals));};
 }
 game.packs.set('xdy-pf2e-workbench.asymonous-benefactor-macros-internal',{getDocuments:async()=>[macro]});
 let valid=true;const operation=createMedicNative({game,canvas,Dialog,ChatMessage,Hooks,CONFIG:{Dice:{rolls:[DamageRoll,CheckRoll]}}})({actor,healer,target,user:{id:'owner'},branch:'battle-medicine',continuation:{actorUuid:actor.uuid,cardId:'original',nonce:'nonce'},validate(){if(!valid)throw Error('expired');}});
 return {operation,observed,cards,game,canvas,Hooks,hooks,release,invalidate:()=>valid=false};
}
const flush=()=>new Promise(resolve=>setImmediate(resolve));
const html={find:()=>({val(){return this;},prop(){return this;},trigger(){return this;}})};
const wbPath=process.env.FVTT_WORKBENCH_MACRO??'C:/Users/Taka/Desktop/fvtt/tmp/team-automation-20260917/coverage-evidence/live-Workbench-Treat-Wounds-and-Battle-Medicine.mjs';
test('installed Workbench reads a locked native canvas layer and waits for its roll and Dice So Nice result', {skip:!existsSync(wbPath)},async()=>{
 const f=detachedWorkbench({macroSource:readFileSync(wbPath,'utf8')});let done=false;f.operation.then(()=>done=true);await flush();
 const nodes={useBattleMedicine:{value:'1'},'dc-type':{value:'1'},modifier:{value:'0'}};
 const form={find(selector){const node=nodes[selector.match(/name="([^"]+)"/)?.[1]];return {0:node,length:node?1:0,val(value){if(value===undefined)return node?.value;if(node)node.value=value;return this;},prop(){return this;},trigger(){return this;}};}};
 await f.observed.dialog.options.buttons.yes.callback(form);await flush();assert.equal(done,false);assert.ok(f.observed.check);
 // PF2e CheckContext copies extra options into both contextual actors. The native statistic already
 // supplies healer options; forwarding Workbench's self:* options would make the patient an elf too.
 assert.deepEqual(f.observed.check.extraRollOptions,['action:treat-wounds',`${M}:medic-workbench:nonce`]);
 f.release();await flush();assert.equal(done,false);assert.equal(f.cards.length,0);f.Hooks.call('diceSoNiceRollComplete');const result=await f.operation;
 assert.equal(result.status,'delegated');assert.equal(f.cards.length,1);assert.equal(f.cards[0].flags.treat_wounds_battle_medicine.healing,8);assert.equal(f.cards[0].flags[M].medicWorkbench.nonce,'nonce');
});
test('installed unchanged Workbench Assurance binds its actual roll card before the treatment result',{skip:!existsSync(wbPath)},async()=>{
 const f=detachedWorkbench({macroSource:readFileSync(wbPath,'utf8')});await flush();const nodes={useBattleMedicine:{value:'1'},'dc-type':{value:'1'},modifier:{value:'0'},assurance_bool:{checked:true}};
 const form={find(selector){const node=nodes[selector.match(/name="([^"]+)"/)?.[1]];return {0:node,length:node?1:0,val(value){if(value===undefined)return node?.value;if(node)node.value=value;return this;},prop(){return this;},trigger(){return this;}};}};
 await f.observed.dialog.options.buttons.yes.callback(form);assert.equal((await f.operation).status,'delegated');assert.equal(f.cards.length,2);assert.equal(f.cards[0].roll[0].total,22);assert.equal(f.cards[1].flags[M].medicWorkbench.checkId,f.cards[0].id);assert.equal(f.observed.check,undefined);
});
test('detached Workbench callback waits for exact native check and delayed healing card; pinned selections remain local',async()=>{
 const f=detachedWorkbench();let finished=false;f.operation.then(()=>finished=true);await flush();await f.observed.dialog.options.buttons.yes.callback(html);await flush();assert.equal(finished,false);
 assert.equal(f.observed.scope.canvas.tokens.controlled[0].actor.uuid,'Actor.healer');assert.equal([...f.observed.scope.game.user.targets][0].id,'t');assert.equal(f.game.user.targets.values().next().value.id,'wrong');assert.equal(f.canvas.tokens.controlled[0].id,'wrong');
 f.release();await flush();assert.equal(finished,false);await f.observed.healReady();assert.equal((await f.operation).status,'delegated');assert.equal(f.cards[0].flags[M].medicWorkbench.nonce,'nonce');assert.equal(f.observed.check.extraRollOptions.includes(`${M}:medic-workbench:nonce`),true);assert.equal(f.hooks.size,0);
});
test('detached native Workbench check cancellation closes continuation without a healing card',async()=>{const f=detachedWorkbench({cancel:true});await flush();await f.observed.dialog.options.buttons.yes.callback(html);f.release();assert.equal((await f.operation).status,'cancelled');assert.equal(f.cards.length,0);assert.equal(f.hooks.size,0);});
test('Workbench revalidates at native modifier submission and blocks stale treatment',async()=>{const f=detachedWorkbench();const result=assert.rejects(f.operation,/expired/);await flush();await f.observed.dialog.options.buttons.yes.callback(html);f.invalidate();let prevented=false;f.observed.submit({preventDefault(){prevented=true;},stopImmediatePropagation(){}});await result;assert.equal(prevented,true);assert.equal(f.cards.length,0);assert.equal(f.hooks.size,0);});
test('Workbench validates immediately before native submission and propagates cancellation',async()=>{
 const actor={},healer={object:{actor}},target={object:{actor:{}}},game={user:{targets:new Set()},modules:new Map([['xdy-pf2e-workbench',{active:true}]]),packs:new Map()},calls=[];
 class Dialog{constructor(options){this.options=options;}}
 game.packs.set('xdy-pf2e-workbench.asymonous-benefactor-macros-internal',{getDocuments:async()=>[{execute:async scope=>{const d=new scope.Dialog({buttons:{yes:{callback:()=>calls.push('applied')},no:{}},render:()=>{}});await d.options.buttons.no.callback();}}]});
 const result=await createMedicNative({game,canvas:{tokens:{}},Dialog})({actor,healer,target,branch:'battle-medicine',validate:()=>{}});
 assert.equal(result.status,'cancelled');assert.deepEqual(calls,[]);
});
