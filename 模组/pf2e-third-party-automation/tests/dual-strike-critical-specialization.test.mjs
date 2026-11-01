import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {buildDualStrikeCriticalCompatibility} from '../scripts/dual-strike-critical-compat.mjs';
const {createDualStrikeAutomation}=await import(process.env.DUAL_STRIKE_UNDER_TEST??'../scripts/dual-strike-automation.mjs');

const ID='pf2e-third-party-automation';
const source=fs.readFileSync(process.env.PF2E_NATIVE_861_BUNDLE??'C:/Users/Taka/Desktop/fvtt/work/bob2-tpa-gap-research-20261101/snapshot/system/pf2e.mjs','utf8');
const match=source.match(/Wn = (class Predicate[\s\S]*?), AutomaticBonusProgression\$1 =/);assert(match);
const Predicate=Function('M',`let Wn=${match[1]};return Wn;`)(x=>!!x&&typeof x==='object'&&!Array.isArray(x));
const swordPredicate=['check:outcome:critical-success','critical-specialization',{or:['item:group:sword','crit-item-1:group:sword','crit-item-2:group:sword']}];
const captured='C:/Users/Taka/Desktop/fvtt/work/bob2-tpa-gap-research-20261101';
const tb=fs.readFileSync(process.env.TOOLBELT_3592_SOURCE??`${captured}/snapshot/modules/pf2e-toolbelt/scripts/main.js`,'utf8');
const pt=fs.readFileSync(process.env.PATREON_3230_SOURCE??`${captured}/snapshot/modules/patreon-v3/src/index.js`,'utf8');
const core=fs.readFileSync(process.env.FOUNDRY_14369_ROLL_SOURCE??'C:/Users/Taka/Desktop/fvtt/work/bob2-tpa-implementation-20261101/runtime/core/client/dice/roll.mjs','utf8');
const section=(text,start,end)=>{const a=text.indexOf(start),b=text.indexOf(end,a+start.length);assert(a>=0&&b>a,start);return text.slice(a,b);};
const nativeToMessage=Function(`return ({${section(core,'  async toMessage(messageData=','\n  /* -------------------------------------------- */')}}).toMessage;`)();
const mergeCode=['ra','dv','Jl','fv','gD','hD'].map(name=>section(tb,`${name==='Jl'?'async ':''}function ${name}(`,`a(${name},`)).join('\n');
const libs={unique:a=>[...new Set(a)],filter:(a,fn)=>fn?a.filter(fn):list=>list.filter(a),isTruthy:Boolean,sumBy:(a,fn)=>a.reduce((n,item)=>n+fn(item),0),pipe:(x,...fns)=>fns.reduce((v,fn)=>fn(v),x),join:separator=>list=>list.join(separator)};
const nativeMerge=Function('E','u','uv','mv','N','Tt','Ii','getDocumentClass',mergeCode+';return Jl;')({id:'pf2e'},libs,()=>undefined,{parse:x=>x},()=>({querySelector:()=>null}),()=>[],{setMessageFlagTargets:()=>{}},()=>globalThis.CONFIG.ChatMessage.documentClass);
const messageType=Function(section(pt,'function H(a)','r(H,')+';return H;')();
const nativeOptions=Function('H',section(pt,'function ce(a,e)','r(ce,')+';return ce;')(messageType);
const assign=(object,changes)=>{for(const [key,value]of Object.entries(changes)){const parts=key.split('.');let at=object;for(const part of parts.slice(0,-1))at=at[part]??={};at[parts.at(-1)]=value;}};
function roll(total=10){const term={class:'NumericTerm',number:total,options:{},evaluated:true,expression:String(total)};return {total,_evaluated:true,instances:[{type:'slashing',persistent:false,total,materials:new Set(),head:term,terms:[term],componentTotal:()=>0,toJSON:()=>({terms:[structuredClone(term)]})}],options:{},toJSON(){return {total:this.total,options:structuredClone(this.options)};},toMessage:nativeToMessage};}
async function fixture(t,{outcomes=['criticalSuccess','success'],groups=['sword','hammer'],specializations=[true,true],world='bob-2',remote=false,patreon=true,trove=false,twin=false}={}){
 const prior=['CONFIG','game','foundry'].map(key=>[key,globalThis[key]]);t.after(()=>{for(const [key,value]of prior)globalThis[key]=value;});
 const messages=[],requests=[];
 class Message{
  constructor(data){const serialized={content:'',...data,rolls:(data.rolls??[]).map(r=>r.toJSON?.()??r)};Object.assign(this,serialized);this._source=structuredClone(serialized);this.rolls=serialized.rolls.map(data=>Object.assign(roll(data.total),data));}
  applyMode(){}
  toObject(){return structuredClone(this._source);}
  static async create(data){const message=new Message({...data,id:`m${messages.length}`});messages.push(message);game.messages.set(message.id,message);return message;}
 }
 globalThis.CONFIG={sounds:{dice:'dice'},ChatMessage:{documentClass:Message},Dice:{rolls:[{name:'DamageRoll',fromData:data=>Object.assign(roll(data.total),data)}]}};
 globalThis.foundry={utils:{mergeObject:(a,b)=>({...a,...b}),getDocumentClass:()=>Message}};
 const actor={id:'pc',uuid:'Actor.pc',signature:'hunter',flags:{},type:'character',items:[],testUserPermission:()=>true,update:async function(changes){assign(this,changes);},system:{actions:[]}};
 const feat={id:'double',type:'feat',sourceId:twin?'Compendium.pf2e.feats-srd.Item.Gw0wGXikhAhiGoud':'Compendium.pf2e.feats-srd.Item.onde0SxLoxLBTnvm',actor,system:{rules:[{key:'FlatModifier',predicate:[{or:['double-slice-second',`${ID}:double-slice-second`]}]}]}};actor.items=[feat];
 const scene={id:'scene',tokens:new Map()},target={id:'target',uuid:'Scene.scene.Token.target',documentName:'Token',parent:scene,actor:{uuid:'Actor.target'},object:{}},origin={id:'origin',uuid:'Scene.scene.Token.origin',documentName:'Token',parent:scene,actor};
 scene.tokens.set(target.id,target);scene.tokens.set(origin.id,origin);actor.getActiveTokens=()=>[origin];
 target.actor.getRollOptions=()=>['self:prey:hunter'];
 const weapons=groups.map((group,index)=>({id:`w${index}`,uuid:`Actor.pc.Item.w${index}`,name:`Weapon ${index}`,group,type:'weapon',isMelee:true,system:{equipped:{carryType:'held',handsHeld:1},traits:{value:[]}}}));
 actor.system.actions=weapons.map(item=>({type:'strike',item}));
 const mergeOwner={render:async()=>'',deleteFlagProperty:()=>{},getMessageTargets:()=>[],setFlagProperties(data,key,value){data.flags['pf2e-toolbelt']??={};data.flags['pf2e-toolbelt'].betterChat??={};data.flags['pf2e-toolbelt'].betterChat[key]=value;}};
 const gm={id:'gm',targets:{first:()=>null}},game={world:{id:world},system:{id:'pf2e'},user:gm,users:{activeGM:gm},i18n:{localize:value=>value},settings:{get:module=>module==='patreon-v3'?{uYv7S8tdeXXpmd8a:{isActive:patreon,handlerRules:[{triggerType:'damage-roll',value:'criticalSpecializationSword'}]}}:module==='trigger-engine'?{enabled:['6JdDvJY9YKdxoi2o'],disabled:[],sources:[]}:'public'},time:{worldTime:100},scenes:new Map([[scene.id,scene]]),messages:new Map(),modules:new Map([['pf2e-toolbelt',{active:true}],['patreon-v3',{active:patreon}],['trigger-engine',{active:trove}],['pf2e-trigger-trove',{active:trove}]]),toolbelt:{api:{betterChat:{mergeDamageMessages:(a,b,options)=>nativeMerge.call(mergeOwner,a,b,options)}}}};globalThis.game=game;
 if(trove==='guarded'){
  game.modules.get('trigger-engine').active=true;game.modules.get('pf2e-trigger-trove').active=true;
  const graph=JSON.parse(fs.readFileSync(`${captured}/snapshot/modules/pf2e-trigger-trove/triggers/base-triggers.json`,'utf8')).find(graph=>graph.id==='6JdDvJY9YKdxoi2o');
  const setting=buildDualStrikeCriticalCompatibility(game.settings.get('trigger-engine'),{graph}),get=game.settings.get;
  game.settings.get=module=>module==='trigger-engine'?setting:get(module);
 }
 const nativeOperations={register(){},async run(_ctx,request){
  requests.push(request);const index=weapons.findIndex(weapon=>weapon.id===request.weaponId);
  if(request.type==='attack'){const check=await Message.create({speaker:{actor:actor.id},flags:{pf2e:{context:{type:'attack-roll',outcome:outcomes[index],options:request.options}}},rolls:[]});return {status:'rolled',messageId:check.id};}
  const damage=roll(request.critical?20:10),context={options:[`item:group:${groups[index]}`,`item:id:w${index}`,...specializations[index]?['critical-specialization']:[]]};
  return {status:'rolled',...remote?{roll:damage.toJSON()}:{nativeRoll:damage},context,privacy:{blind:true,whisper:['gm'],messageMode:'blind'}};
 }};
 const provider=createDualStrikeAutomation({game,nativeOperations,choose:async({choices})=>choices[0].value,fromUuid:async uuid=>[target,origin].find(token=>token.uuid===uuid)});
 const message={id:'usage',speaker:{actor:actor.id},flags:{[ID]:{usageInput:{targetUuids:[target.uuid]}}}};
 const use=()=>provider.executeUsage({actor,item:feat,message,user:gm,action:twin?'dual:twin-takedown':'dual:double-slice'});
 await use();const card=messages.find(m=>m.flags[ID]?.dualStrike);
 return {card,messages,requests,use,weapons,game,options:nativeOptions(card)};
}

for(const remote of [false,true])for(const [outcomes,expected]of [
 [['criticalSuccess','success'],['Actor.pc.Item.w0']],
 [['success','criticalSuccess'],['Actor.pc.Item.w1']],
 [['criticalSuccess','criticalSuccess'],['Actor.pc.Item.w0','Actor.pc.Item.w1']],
 [['criticalSuccess','failure'],['Actor.pc.Item.w0']],
])test(`Double Slice retains each sword critical specialization (${outcomes}, ${remote?'serialized owner':'local roll'})`,async t=>{
 const f=await fixture(t,{outcomes,groups:['sword','sword'],remote});
 assert.equal(Predicate.test(swordPredicate,f.options),true);
 assert.deepEqual(f.card.flags[ID].dualStrike.criticalSpecializations.map(hit=>hit.weaponUuid),expected);
 assert.equal(f.card.blind,true);assert.deepEqual(f.card.whisper,['gm']);assert.equal(f.card.flags.pf2e.context.target.token,'Scene.scene.Token.target');
 assert.equal(f.messages.filter(m=>m.flags[ID]?.dualStrike).length,1);
 await f.use();assert.equal(f.messages.filter(m=>m.flags[ID]?.dualStrike).length,1);
});
for(const setup of [
 {outcomes:['success','criticalSuccess'],groups:['sword','hammer'],specializations:[true,true]},
 {outcomes:['criticalSuccess','criticalSuccess'],groups:['sword','hammer'],specializations:[false,true]},
 {outcomes:['success','success'],groups:['sword','sword'],specializations:[true,true]},
])test(`another hit cannot supply sword critical specialization: ${JSON.stringify(setup)}`,async t=>{
 const f=await fixture(t,setup);assert.equal(Predicate.test(swordPredicate,f.card.flags.pf2e.context.options),false);
});
test('second sword critical is attributed to that weapon without replacing the damage origin',async t=>{
 const f=await fixture(t,{outcomes:['success','criticalSuccess'],groups:['hammer','sword']});
 assert.equal(Predicate.test(swordPredicate,f.card.flags.pf2e.context.options),true);
 assert.equal(f.card.flags.pf2e.origin.uuid,'Actor.pc.Item.w0');
 assert.deepEqual(f.card.flags[ID].dualStrike.criticalSpecializations.map(hit=>hit.weaponUuid),['Actor.pc.Item.w1']);
});
test('other worlds retain the existing Double Slice damage flow',async t=>{
 const f=await fixture(t,{world:'-'});assert.equal(f.card.rolls[0].total,30);
 assert.equal(f.card.flags[ID].dualStrike.criticalSpecializations,undefined);assert.equal(Predicate.test(swordPredicate,f.card.flags.pf2e.context.options),false);
});
for(const options of [{patreon:false},{trove:true},{twin:true}])test(`sword specialization is not restored outside the repaired activity and consumer: ${JSON.stringify(options)}`,async t=>{
 const f=await fixture(t,options);assert.equal(Predicate.test(swordPredicate,f.options),false);assert.equal(f.card.flags[ID].dualStrike.criticalSpecializations,undefined);
});
test('native damage application passes the marked merged card to the Trove exclusion',async t=>{
 const f=await fixture(t,{trove:'guarded',outcomes:['success','criticalSuccess'],groups:['hammer','sword']});
 assert.equal(Predicate.test(swordPredicate,f.options),true);
 let applied;
 const target={alliance:'opposition',getSelfRollOptions:(domain='self')=>[`${domain}:type:npc`],getContextualClone(){return this;},async applyDamage(params){applied=params;}};
 const token={actor:target,flags:{pf2e:{}}};class DamageRoll{total=30;alter(){return this;}}
 const apply=Function('htmlQuery','ui','game','CONFIG','un','ErrorPF2e','_t','__nativeManualPoolBatch','extractEphemeralEffects','toggleOffShieldBlock','shiftAdjustDamage',section(source,'async function applyDamageFromMessage({ message: e','\nasync function shiftAdjustDamage(')+';return applyDamageFromMessage;')(()=>null,{chat:{element:{}},notifications:{error(){throw Error('no token');}}},{user:{getActiveTokens:()=>[token]}},{PF2E:{chatDamageButtonShieldToggle:false}},DamageRoll,Error,list=>list,{admit:()=>null},async()=>[],()=>{},()=>{throw Error('unexpected prompt');});
 await apply({message:{...f.card,actor:{alliance:'party'},item:{isOfType:()=>false},rolls:[new DamageRoll()]}});
 const graph=f.game.settings.get('trigger-engine').sources[0],trovePredicate=JSON.parse(graph.nodes.find(node=>node.id==='oYYR4W7IaJpI1OpI').inputs.predicate.value);
 assert(applied.rollOptions.has(`${ID}:dual-strike-critical-specialization`));assert.equal(Predicate.test(trovePredicate,applied.rollOptions),false);
 assert.equal(Predicate.test(trovePredicate,['check:outcome:critical-success','critical-specialization','item:group:sword']),true);
});
