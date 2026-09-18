import test from 'node:test';
import assert from 'node:assert/strict';
import {createDesperatePrayerProvider,PRAYER_SOURCES as S} from '../scripts/desperate-prayer.mjs';
import {MODULE_ID as ID} from '../scripts/rules.mjs';
import {SerialActions} from '../scripts/runtime.mjs';
import {createNativeCastEvents} from '../scripts/amp-cast-events.mjs';
import {getNativeActionEvents} from '../scripts/native-action-events.mjs';
import {installActionEntrances} from '../scripts/metapower/entrances.mjs';

function fixture({focus=0,decision='use',actions,beforeRegister}={}){
 const gm={id:'gm',active:true,isGM:true},player={id:'player',active:true},users=new Map([[gm.id,gm],[player.id,player]]);users.activeGM=gm;
 const hooks=new Map(),wrappers=new Map(),rpc=new Map(),updates=[],errors=[],docs=new Map();
 const Hooks={on(k,fn){const rows=hooks.get(k)??[];rows.push(fn);hooks.set(k,rows);return fn},off(){}};
 const emit=(k,...args)=>{for(const fn of hooks.get(k)??[])if(fn(...args)===false)return false};
 const assign=(doc,changes)=>{for(const[p,v]of Object.entries(changes)){let o=doc;const keys=p.split('.');for(const k of keys.slice(0,-1))o=o[k]??={};o[keys.at(-1)]=structuredClone(v)}};
 const actor={id:'pc',uuid:'Actor.pc',type:'character',canAct:true,isDead:false,flags:{},items:new Map(),system:{resources:{focus:{value:focus,max:3}}},testUserPermission:u=>u===gm||u===player};
 const persist=async(changes,options={})=>{if(emit('preUpdateActor',actor,changes,options,game.user.id)===false)return actor;updates.push(structuredClone(changes));assign(actor,changes);emit('updateActor',actor,changes,options,game.user.id);return actor};
 actor.update=(changes,options={})=>{const fn=wrappers.get('CONFIG.Actor.documentClass.prototype.update');return fn?fn.call(actor,persist,changes,options):persist(changes,options)};
 const item=(id,source,type='feat',system={})=>{const i={id,uuid:`Actor.pc.Item.${id}`,type,sourceId:source,actor,system,flags:{},async update(changes,options={}){emit('preUpdateItem',i,changes,options,player.id);assign(i,changes);emit('updateItem',i,changes,options,player.id);return i}};actor.items.set(id,i);docs.set(i.uuid,i);return i};
 const feat=item('prayer',S.prayer,'feat',{frequency:{value:1,max:1,per:'day'}}),devotion=item('devotion',S.devotion),domain=item('domain',S.domain,'feat',{rules:[{key:'ChoiceSet',flag:'deitysDomain',selection:'zeal'}]});
 const entry=item('entry','entry','spellcastingEntry',{prepared:{value:'focus'}});
 const spell=(id,source)=>{const i=item(id,source,'spell',{location:{value:'entry'},cast:{focusPoints:1},traits:{value:['focus']}});i.spellcasting=entry;return i};
 const lay=spell('lay',S.lay),surge=spell('surge',S.surge),other=spell('other','Compendium.test.other');
 const combat={id:'combat',uuid:'Combat.combat',started:true,round:1,turn:0,turns:[]},token={uuid:'Scene.scene.Token.token',actor,parent:{id:'scene'}};
 const combatant={id:'combatant',uuid:'Combat.combat.Combatant.combatant',actor,token,encounter:combat,flags:{pf2e:{roundOfLastTurn:1}}};combat.combatant=combatant;combat.turns=[combatant];
 const game={user:gm,users,actors:new Map([[actor.id,actor]]),messages:new Map(),combats:new Map([[combat.id,combat]]),combat,modules:new Map(),pf2e:{actions}};for(const d of [actor,combat,combatant,token])docs.set(d.uuid,d);
 const q=new SerialActions(),castEvents={withActorResourceLock:(a,fn)=>q.run(a.uuid,fn)};let provider,serial=0,payment,confirmImpl=async()=>decision;
 async function pay(){const win=actor.flags[ID].desperatePrayer.window,proof={id:`frequency${++serial}`,itemUuid:feat.uuid,userId:player.id,before:1,after:0};
  const options={[ID]:{frequencyReceipt:proof}};await feat.update({'system.frequency.value':0},options);
  const message={id:`message${serial}`,uuid:`ChatMessage.message${serial}`,author:player,speaker:{actor:actor.id},rolls:[],flags:{pf2e:{origin:{uuid:feat.uuid,actor:actor.uuid,type:'feat'}},[ID]:{usageInput:{actualUse:true,frequencyReceiptId:proof.id},...provider.captureUsage(feat)}}};game.messages.set(message.id,message);
  payment={actor,item:feat,message,user:player,action:'desperate-prayer:use',frequencyReceipt:proof};return provider.executeUsage(payment);
 }
 beforeRegister?.(game);
 provider=createDesperatePrayerProvider({game,castEvents,fromUuid:async u=>docs.get(u),choose:c=>confirmImpl(c),useOriginal:pay,onError:e=>errors.push(e),randomId:()=>`nonce${++serial}`});const dispose=provider.register({Hooks,libWrapper:{register:(_id,p,fn)=>wrappers.set(p,fn),unregister:(_id,p)=>wrappers.delete(p)},socket:{register:(k,fn)=>rpc.set(k,fn)}});
 async function consume(which=lay,{cost=1,result=true,replyLost=false,deferred=false}={}){which.system.cast.focusPoints=cost;let commit;const context={actor,item:which,entry,user:player,payload:{id:`cast${++serial}`,focusPoints:cost},expectFocusCommit:spec=>{commit=spec}};
  const native=async()=>{if(!result)return false;const before=actor.system.resources.focus.value;if(before<cost)return false;const proof={castNonce:context.payload.id,itemUuid:which.uuid,entryUuid:entry.uuid,before,after:before-cost,cost};const extra=commit&&!deferred?commit.changes(proof):{};await actor.update({'system.resources.focus.value':before-cost,...extra});if(replyLost)throw Error('reply lost');return true};return provider.consumePolicy(context,native);
 }
 return {game,actor,feat,devotion,domain,lay,surge,other,entry,combat,combatant,provider,dispose,updates,errors,hooks,wrappers,rpc,pay,consume,emit,get payment(){return payment},confirm:fn=>confirmImpl=fn,start:()=>provider.onStartTurn(combatant),end:async()=>{combat.turn=1;combat.combatant=null;return provider.onEndTurn(combatant)}};
}
test('exact original source only; ordinary display and arbitrary own-turn Use cannot grant',()=>{const f=fixture();assert.equal(f.provider.resolveAction(f.feat),'desperate-prayer:use');assert.equal(f.provider.resolveAction({...f.feat,sourceId:'wrong'}),undefined);assert.throws(()=>f.provider.beforeUse(f.feat,f.game.users.get('player')),/起回合/);assert.equal(f.actor.system.resources.focus.value,0)});
test('native start opportunity pays original frequency and atomically grants exactly one',async()=>{const f=fixture();await f.start();assert.equal(f.feat.system.frequency.value,0);assert.equal(f.actor.system.resources.focus.value,1);assert.equal(f.actor.flags[ID].desperatePrayer.credit.state,'available');const grant=f.updates.find(c=>c['system.resources.focus.value']===1);assert(grant[`flags.${ID}.desperatePrayer`]);await f.provider.executeUsage(f.payment);await f.start();assert.equal(f.actor.system.resources.focus.value,1)});
test('declining and starting with normal Focus do not leave a later opportunity',async()=>{for(const options of [{decision:'skip'},{focus:1}]){const f=fixture(options);await f.start();f.actor.system.resources.focus.value=0;assert.throws(()=>f.provider.beforeUse(f.feat),/起回合/);assert.equal(f.feat.system.frequency.value,1)}});
test('changing turn while owner answers cannot grant or consume daily use',async()=>{const f=fixture();f.confirm(async()=>{f.combat.turn=1;f.combat.combatant=null;return 'use'});await f.start();assert.equal(f.actor.system.resources.focus.value,0);assert.equal(f.feat.system.frequency.value,1)});
test('ordinary action attempted before choice closes the start opportunity',async()=>{const f=fixture();f.confirm(async()=>{await f.provider.beforeAction(f.actor);return 'use'});await f.start();assert.equal(f.feat.system.frequency.value,1);assert.equal(f.actor.system.resources.focus.value,0)});
for(const key of ['lay','surge'])test(`native ${key} payment consumes temporary credit and never consumes twice`,async()=>{const f=fixture();await f.start();assert.equal(await f.consume(f[key]),true);assert.equal(f.actor.system.resources.focus.value,0);assert.equal(f.actor.flags[ID].desperatePrayer.credit.state,'spent');await f.end();assert.equal(f.actor.system.resources.focus.value,0)});
test('Weapon Surge requires the exact current zeal grant, not a cleric or champion trait guess',async()=>{const f=fixture();await f.start();f.domain.system.rules[0].selection='fire';await assert.rejects(f.consume(f.surge),/虔诚|普通/);assert.equal(f.actor.system.resources.focus.value,1)});
test('unrelated focus spell cannot spend temporary-only credit',async()=>{const f=fixture();await f.start();await assert.rejects(f.consume(f.other),/虔诚|普通/);assert.equal(f.actor.system.resources.focus.value,1)});
test('later ordinary points can pay unrelated spells; expiry removes only remaining temporary point',async()=>{const f=fixture();await f.start();await f.actor.update({'system.resources.focus.value':2});await f.consume(f.other);assert.equal(f.actor.system.resources.focus.value,1);assert.equal(f.actor.flags[ID].desperatePrayer.credit.remaining,1);await f.end();assert.equal(f.actor.system.resources.focus.value,0);await f.end();assert.equal(f.actor.system.resources.focus.value,0)});
test('devotion payment spends temporary first and preserves later ordinary point at expiry',async()=>{const f=fixture();await f.start();await f.actor.update({'system.resources.focus.value':2});await f.consume(f.lay);await f.end();assert.equal(f.actor.system.resources.focus.value,1)});
test('unspent temporary point expires without touching later ordinary points',async()=>{const f=fixture();await f.start();await f.actor.update({'system.resources.focus.value':3});await f.end();assert.equal(f.actor.system.resources.focus.value,2)});
test('zero-cost and rejected native payments leave the temporary point unspent',async()=>{const f=fixture();await f.start();await f.consume(f.lay,{cost:0});assert.equal(f.actor.flags[ID].desperatePrayer.credit.remaining,1);await f.consume(f.lay,{result:false});assert.equal(f.actor.flags[ID].desperatePrayer.credit.remaining,1)});
test('native atomic proof survives lost reply; expiry cannot remove a later ordinary point',async()=>{const f=fixture();await f.start();await f.actor.update({'system.resources.focus.value':2});await assert.rejects(f.consume(f.lay,{replyLost:true}),/reply lost/);assert.equal(f.actor.flags[ID].desperatePrayer.credit.state,'spent');await f.end();assert.equal(f.actor.system.resources.focus.value,1)});
test('unbound/deferred debit and subsequent ordinary recovery remain uncertain, never auto-expired',async()=>{const f=fixture();await f.start();await assert.rejects(f.consume(f.lay,{deferred:true}),/确认|绑定|不确定/);await f.actor.update({'system.resources.focus.value':1});await f.end();assert.equal(f.actor.system.resources.focus.value,1);assert.equal(f.actor.flags[ID].desperatePrayer.credit.state,'uncertain')});
test('manual downward overwrite cannot be guessed as consuming a specific credit',async()=>{const f=fixture();await f.start();await f.actor.update({'system.resources.focus.value':0});await f.actor.update({'system.resources.focus.value':1});await f.end();assert.equal(f.actor.system.resources.focus.value,1);assert.equal(f.actor.flags[ID].desperatePrayer.credit.state,'uncertain')});
test('non-owner, inactive GM and forged display fail without resource changes',async()=>{const f=fixture();await f.start();const old=f.actor.system.resources.focus.value;for(const c of [{...f.payment,user:{id:'stranger'}},{...f.payment,message:{...f.payment.message,id:'copied'}}])await assert.rejects(f.provider.executeUsage(c));f.game.user=f.game.users.get('player');await assert.rejects(f.provider.executeUsage(f.payment));assert.equal(f.actor.system.resources.focus.value,old)});
test('check middleware closes the window without registering the shared Check.roll path twice',async()=>{const f=fixture();assert.equal(f.wrappers.has('game.pf2e.Check.roll'),false);assert.equal(await f.provider.interceptCheck(()=>7,{},{}),7);f.confirm(async()=>{await f.provider.interceptCheck(()=>7,{},{actor:f.actor});return 'use'});await f.start();assert.equal(f.feat.system.frequency.value,1)});
test('removing Prayer does not release an already granted restricted point to unrelated spells',async()=>{const f=fixture();await f.start();f.actor.items.delete(f.feat.id);assert.equal(f.provider.isManagedActor(f.actor),true);await assert.rejects(f.consume(f.other),/虔诚|普通/)});
test('late old-GM opportunity cannot be used after active GM changes',async()=>{const f=fixture();let rejected=false;f.confirm(async()=>{f.game.users.activeGM={id:'newgm'};try{f.provider.beforeUse(f.feat,f.game.users.get('player'))}catch(e){rejected=/主GM|起回合/.test(e.message)}return 'skip'});await assert.rejects(f.start(),/主GM/);assert.equal(rejected,true);assert.equal(f.feat.system.frequency.value,1)});
test('real cast coordinator and Prayer combine the original native focus debit with its credit receipt',async()=>{
 const f=fixture();await f.start();const casts=createNativeCastEvents({game:f.game,fromUuid:async uuid=>[f.actor,f.entry,f.lay].find(d=>d.uuid===uuid)});
 casts.addActorMatcher(f.provider.isManagedActor);casts.addConsumePolicy(f.provider.consumePolicy);
 casts.register({libWrapper:{register:(_id,p,fn)=>f.wrappers.set(p,fn)},socket:{register:(k,fn)=>f.rpc.set(k,fn)}});
 f.lay.rank=1;
 f.entry.consume=(item,rank,slot,cap)=>f.wrappers.get('CONFIG.PF2E.Item.documentClasses.spellcastingEntry.prototype.consume').call(f.entry,async()=>{await f.actor.update({'system.resources.focus.value':f.actor.system.resources.focus.value-item.system.cast.focusPoints});return true},item,rank,slot,cap);
 const payload={id:'nativecast',actorUuid:f.actor.uuid,itemUuid:f.lay.uuid,sourceId:f.lay.sourceId,entryUuid:f.entry.uuid,rank:1,slotId:null,focusPoints:1,overlayIds:[]};
 const pay=()=>f.rpc.get('native-cast-pay').call({socketdata:{userId:'player'}},payload);
 assert.equal((await pay()).ok,true);assert.equal(f.actor.system.resources.focus.value,0);assert.equal(f.actor.flags[ID].desperatePrayer.credit.state,'spent');
 const debit=f.updates.filter(c=>c['system.resources.focus.value']===0);assert.equal(debit.length,1);assert.equal(debit[0][`flags.${ID}.desperatePrayer`].credit.payments[0].castNonce,'nativecast');
 assert.equal((await pay()).ok,true);assert.equal(f.updates.filter(c=>c['system.resources.focus.value']===0).length,1);
});
test('native basic action Use closes a pending start decision before the action executes',async()=>{
 const f=fixture();let acted=false,offered=false;
 class Variant{constructor(){this.slug='raise-a-shield'}async use(){acted=true;return 'native-action'}}
 const action={toActionVariant:()=>new Variant(),variants:[]};f.game.pf2e.actions=new Map([['raise-a-shield',action]]);
 // register once on a fresh provider after exposing a native action collection.
 const p=createDesperatePrayerProvider({game:f.game,castEvents:{withActorResourceLock:(_a,fn)=>fn()},fromUuid:async()=>null,choose:async()=>{offered=true;assert.equal(await action.toActionVariant().use({actors:[f.actor]}),'native-action');return 'use'},useOriginal:()=>{throw Error('must not use Prayer after another action')},randomId:()=> 'actionwindow'});
 p.register({Hooks:{on(){},off(){}},libWrapper:{register(){},unregister(){}}});await p.onStartTurn(f.combatant);assert.equal(offered,true);assert.equal(acted,true);assert.equal(f.feat.system.frequency.value,1);
});
test('silent real cast closes the opportunity; consume:false preview does not',async()=>{
 for(const preview of [false,true]){
  const f=fixture();let called=0;f.confirm(async()=>{await f.provider.interceptCast({item:f.other,options:{message:false,consume:!preview}},()=>{called++;return 7});return 'use'});
  await f.start();assert.equal(called,1);assert.equal(f.actor.system.resources.focus.value,preview?1:0);
 }
});
test('changed daily frequency schema cannot reuse a paid Prayer message',async()=>{const f=fixture();await f.start();f.feat.system.frequency.per='hour';await assert.rejects(f.provider.executeUsage(f.payment),/次数|每日/)});

test('Foundry-expanded atomic Prayer receipt is not overwritten as an unexplained focus decrease',async()=>{
 const f=fixture();await f.start();const credit=f.actor.flags[ID].desperatePrayer.credit;
 const changes={system:{resources:{focus:{value:0}}},flags:{[ID]:{desperatePrayer:{credit:{...credit,state:'spent',remaining:0,totalObserved:0,payments:[{castNonce:'paid'}]}}}}};
 f.emit('preUpdateActor',f.actor,changes,{},'gm');
 assert.equal(changes[`flags.${ID}.desperatePrayer`],undefined);assert.equal(changes.flags[ID].desperatePrayer.credit.state,'spent');assert.equal(changes.flags[ID].desperatePrayer.credit.payments[0].castNonce,'paid');
});

test('an already cached metapower variant closes Prayer before the complete original native action',async t=>{
 let f,removeMeta;const calls=[];
 class Variant{async use(){calls.push('native');assert.equal(f.actor.flags[ID].desperatePrayer.window.state,'closed');return 'original'}}
 const cached=new Variant(),action={slug:'sustain',variants:new Map([['cached',cached]]),toActionVariant:()=>new Variant()};
 f=fixture({actions:new Map([['sustain',action]]),beforeRegister:game=>{removeMeta=installActionEntrances({game,eligible:()=>true,observe:async(_s,next)=>{calls.push('meta');return next()}})}});
 t.after(()=>{f.dispose();removeMeta()});f.confirm(async()=>{assert.equal(await cached.use({actors:[f.actor]}),'original');return 'use'});
 await f.start();assert.deepEqual(calls,['meta','native']);assert.equal(f.feat.system.frequency.value,1);assert.equal(f.actor.system.resources.focus.value,0);
});

test('Prayer and a second action observer share one scope; Prayer unsubscribe leaves that observer working',async t=>{
 class Variant{async use(){return 'original'}}
 const action={slug:'sustain',variants:[],toActionVariant:()=>new Variant(),use(params){return this.toActionVariant().use(params)}},f=fixture({actions:new Map([['sustain',action]])}),events=getNativeActionEvents({game:f.game}),seen=[];
 const remove=events.addMiddleware((scope,next)=>{seen.push({scope,state:f.actor.flags[ID].desperatePrayer.window.state});return next()});events.register();t.after(()=>{remove();events.cleanup()});
 f.confirm(async()=>{assert.equal(await action.use({actors:[f.actor]}),'original');return 'use'});await f.start();assert.equal(seen[0].state,'closed');assert.equal(seen[0].scope.action,action);assert.equal(f.feat.system.frequency.value,1);
 f.dispose();f.actor.flags[ID].desperatePrayer.window.state='open';assert.equal(await action.use({actors:[f.actor]}),'original');assert.equal(seen.length,2);assert.equal(seen[1].state,'open');assert.equal(f.actor.flags[ID].desperatePrayer.window.state,'open');
});

