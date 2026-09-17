import test from 'node:test';
import assert from 'node:assert/strict';
import {MODULE_ID,SOURCES} from '../scripts/rules.mjs';
let usage={};try{usage=await import('../scripts/usage-events.mjs')}catch(e){if(e.code!=='ERR_MODULE_NOT_FOUND')throw e;}
const fn=name=>{assert.equal(typeof usage[name],'function',`${name} is implemented`);return usage[name]};
const actor={id:'actor',uuid:'Actor.actor',type:'character',testUserPermission:u=>u.id==='player'};
const item=(key='breath')=>({id:'item',uuid:'Actor.actor.Item.item',type:'feat',actor,sourceId:SOURCES[key],system:{frequency:{value:1,max:1}}});
const message=()=>({id:'message',author:{id:'player'},speaker:{actor:'actor'},flags:{pf2e:{origin:{uuid:'Actor.actor.Item.item',actor:'Actor.actor',type:'feat',rollOptions:[]}}},rolls:[]});

test('routes exact original feature cards, including ordinary send to chat',()=>{
 for(const [key,action]of [['breath','breath'],['circadian','rest'],['cycle','cycle']])assert.equal(fn('parseUsageMessage')(message(),item(key)).action,action);
 assert.equal(fn('parseUsageMessage')(message(),{...item(),sourceId:'Compendium.other.Item.fake',name:'鼓舞之息'}),null);
});
test('Toolbelt actual use is the origin roll option, not an invented flag',()=>{
 const m=message();m.flags.pf2e.origin.rollOptions.push('origin:action:slug:use-action');
 assert.equal(fn('parseUsageMessage')(m,item()).actualUse,true);
 assert.equal(fn('parseUsageMessage')(message(),item()).actualUse,false);
});
test('rejects rolls, generated messages and previously consumed messages',()=>{
 for(const mutate of [m=>m.rolls.push({}),m=>m.isCheckRoll=true,m=>m.flags.pf2e.context={type:'skill-check'},m=>m.flags[MODULE_ID]={usageGenerated:true},m=>m.flags[MODULE_ID]={usage:{status:'done'}}]){const m=message();mutate(m);assert.equal(fn('parseUsageMessage')(m,item()),null)}
});
test('rejects mismatched actor/item origins and accepts native self effect cards',()=>{
 const wrong=message();wrong.flags.pf2e.origin.actor='Actor.other';assert.equal(fn('parseUsageMessage')(wrong,item()),null);
 wrong.flags.pf2e.origin.actor=actor.uuid;wrong.flags.pf2e.origin.uuid='Actor.actor.Item.other';assert.equal(fn('parseUsageMessage')(wrong,item()),null);
 const m=message();delete m.flags.pf2e.origin;m.flags.pf2e.context={type:'self-effect',item:'item'};
 assert.equal(fn('parseUsageMessage')(m,item()).action,'breath');
});
test('frequency receipt is based on actual observed decrement and can be claimed once',()=>{
 const t=fn('createFrequencyTracker')({now:()=>1000}),i=item();t.seed(i);
 const proof={id:'r1',itemUuid:i.uuid,userId:'player',before:1,after:0,createdAt:1000};i.system.frequency.value=0;
 assert(t.observe(i,{[MODULE_ID]:{frequencyReceipt:proof}},'player'));
 assert.deepEqual(t.claim('r1',{itemUuid:i.uuid,userId:'player'}),proof);
 assert.throws(()=>t.claim('r1',{itemUuid:i.uuid,userId:'player'}),/回执/);
});
test('concurrent stale 1-to-0 updates do not create a second receipt',()=>{
 const t=fn('createFrequencyTracker')({now:()=>1000}),i=item();t.seed(i);i.system.frequency.value=0;
 const proof={id:'r1',itemUuid:i.uuid,userId:'player',before:1,after:0,createdAt:1000};t.observe(i,{[MODULE_ID]:{frequencyReceipt:proof}},'player');
 assert.equal(t.observe(i,{[MODULE_ID]:{frequencyReceipt:{...proof,id:'r2'}}},'player'),null);
 assert.throws(()=>t.claim('r2',{itemUuid:i.uuid,userId:'player'}),/回执/);
});
test('rejects cross-user cross-item and expired receipts without losing rightful claim',()=>{
 let now=1000;const t=fn('createFrequencyTracker')({now:()=>now,ttl:5000}),i=item();t.seed(i);i.system.frequency.value=0;
 const proof={id:'r1',itemUuid:i.uuid,userId:'player',before:1,after:0,createdAt:1000};t.observe(i,{[MODULE_ID]:{frequencyReceipt:proof}},'player');
 assert.throws(()=>t.claim('r1',{itemUuid:i.uuid,userId:'other'}),/回执/);
 assert.throws(()=>t.claim('r1',{itemUuid:'Actor.other.Item.item',userId:'player'}),/回执/);
 now=7000;assert.throws(()=>t.claim('r1',{itemUuid:i.uuid,userId:'player'}),/回执/);
});
test('unknown metadata or non-decrement never creates a receipt',()=>{
 const t=fn('createFrequencyTracker')(),i=item();t.seed(i);i.system.frequency.value=0;
 assert.equal(t.observe(i,{},'player'),null);
 assert.throws(()=>t.claim('anything',{itemUuid:i.uuid,userId:'player'}));
});
test('a consumed receipt ID cannot be replayed after resetting frequency',()=>{
 const t=fn('createFrequencyTracker')({now:()=>1000}),i=item();t.seed(i);i.system.frequency.value=0;
 const proof={id:'r1',itemUuid:i.uuid,userId:'player',before:1,after:0,createdAt:1000};
 t.observe(i,{[MODULE_ID]:{frequencyReceipt:proof}},'player');t.claim('r1',{itemUuid:i.uuid,userId:'player'});
 i.system.frequency.value=1;t.observe(i,{},'player');i.system.frequency.value=0;
 assert.equal(t.observe(i,{[MODULE_ID]:{frequencyReceipt:proof}},'player'),null);
});

function harness({active=true,initialActors=true,canvas,scenes,requiresActualUse,resolveAction}={}){
 const callbacks=new Map(),Hooks={on:(n,cb)=>{callbacks.set(n,cb);return n},off:n=>callbacks.delete(n)};
 const gm={id:'gm',name:'GM'},player={id:'player'},i=item(),a=i.actor,m=message();a.items=[i];m.update=async changes=>{for(const[k,v]of Object.entries(changes)){if(k===`flags.${MODULE_ID}.usage`)m.flags[MODULE_ID]={...m.flags[MODULE_ID],usage:v}}};
 const game={user:active?gm:player,users:{activeGM:gm,get:id=>id==='player'?player:gm},actors:initialActors?[a]:[],scenes:typeof scenes==='function'?scenes(a):scenes};
 const calls=[],errors=[];const unregister=fn('registerUsageEvents')({game,Hooks,requiresActualUse,resolveAction,canvas:typeof canvas==='function'?canvas(a):canvas,fromUuid:async uuid=>uuid===i.uuid?i:null,executeUsage:async e=>{calls.push(e);return 'ok'},onError:e=>errors.push(e)});
 return {callbacks,game,i,m,calls,errors,unregister};
}
test('active GM handles once and writes a persistent completed marker',async()=>{
 const h=harness();await Promise.all([h.callbacks.get('createChatMessage')(h.m,{},'player'),h.callbacks.get('createChatMessage')(h.m,{},'player')]);
 assert.equal(h.calls.length,1);assert.equal(h.m.flags[MODULE_ID].usage.status,'done');assert.equal(h.calls[0].user.id,'player');
 await h.callbacks.get('createChatMessage')(h.m,{},'player');assert.equal(h.calls.length,1);h.unregister();
});
async function receiveFirstNativeUse(h,id){
 const proof={id,itemUuid:h.i.uuid,userId:'player',before:1,after:0,createdAt:Date.now()};
 h.i.system.frequency.value=0;
 h.callbacks.get('updateItem')(h.i,{'system.frequency.value':0},{[MODULE_ID]:{frequencyReceipt:proof}},'player');
 h.m.flags[MODULE_ID]={usageInput:{actualUse:true,frequencyReceiptId:id}};
 await h.callbacks.get('createChatMessage')(h.m,{},'player');
 return proof;
}
test('first native use without a prior frequency snapshot is rejected',async()=>{
 const h=harness({initialActors:false});
 await receiveFirstNativeUse(h,'no-snapshot');
 assert.equal(h.calls.length,0);assert.equal(h.m.flags[MODULE_ID].usage.status,'error');h.unregister();
});
test('createActor seeds embedded items before the imported actor first native use',async()=>{
 const h=harness({initialActors:false});
 h.callbacks.get('createActor')?.(h.i.actor);
 const proof=await receiveFirstNativeUse(h,'imported-actor');
 assert.equal(h.calls.length,1,'imported actor has a committed before-value');
 assert.deepEqual(h.calls[0].frequencyReceipt,proof);assert.equal(h.m.flags[MODULE_ID].usage.status,'done');h.unregister();
});
test('ready seeds existing synthetic actors on a scene the active GM is not viewing',async()=>{
 const h=harness({initialActors:false,scenes:a=>[{id:'player-scene',tokens:[{actor:a}]}],canvas:{scene:{id:'gm-scene',tokens:[]}}});
 const proof=await receiveFirstNativeUse(h,'different-scene');
 assert.equal(h.calls.length,1);assert.deepEqual(h.calls[0].frequencyReceipt,proof);h.unregister();
});
for(const source of ['current-canvas','canvasReady','createToken'])test(`${source} seeds synthetic actor items before first native use`,async()=>{
 const h=harness({initialActors:false,canvas:source==='current-canvas'?a=>({scene:{tokens:[{actor:a}]}}):undefined});
 if(source==='canvasReady')h.callbacks.get('canvasReady')?.({scene:{tokens:[{actor:h.i.actor}]}});
 if(source==='createToken')h.callbacks.get('createToken')?.({actor:h.i.actor});
 const proof=await receiveFirstNativeUse(h,source);
 assert.equal(h.calls.length,1,`${source} supplies the synthetic actor frequency snapshot`);
 assert.deepEqual(h.calls[0].frequencyReceipt,proof);h.unregister();
});
test('non-active-GM client never executes the same message',async()=>{const h=harness({active:false});await h.callbacks.get('createChatMessage')(h.m,{},'player');assert.equal(h.calls.length,0);h.unregister()});
test('invalid claimed receipt rejects usage instead of silently charging again',async()=>{
 const h=harness();h.m.flags[MODULE_ID]={usageInput:{frequencyReceiptId:'unknown',actualUse:true}};
 await h.callbacks.get('createChatMessage')(h.m,{},'player');assert.equal(h.calls.length,0);assert.equal(h.m.flags[MODULE_ID].usage.status,'error');h.unregister();
});
test('ignores unauthorized speaker and generated survival rolls',async()=>{
 const h=harness();h.m.author={id:'intruder'};await h.callbacks.get('createChatMessage')(h.m,{},'intruder');assert.equal(h.calls.length,0);
 h.m.author={id:'player'};h.m.flags.pf2e.context={type:'skill-check'};await h.callbacks.get('createChatMessage')(h.m,{},'player');assert.equal(h.calls.length,0);h.unregister();
});
test('usage footer renders safe Chinese pending done and failure results',()=>{
 const render=fn('usageStatusHTML');
 assert.match(render({status:'pending'}),/正在自动结算/);
 assert.match(render({status:'done',result:'恢复1点聚能'}),/恢复1点聚能/);
 assert.match(render({status:'error',error:'<script>x<\/script>'}),/&lt;script&gt;/);
 assert.equal(render(undefined),'');
});

test('opted-in display cards are quietly ignored before usage claim, receipt claim or executor',async()=>{
 const h=harness({resolveAction:()=> 'spell-combination:combination',requiresActualUse:(i,action)=>i===h.i&&action==='spell-combination:combination'});
 h.m.flags[MODULE_ID]={usageInput:{actualUse:false,frequencyReceiptId:'not-a-use'}};
 const before=structuredClone(h.m.flags);
 await h.callbacks.get('createChatMessage')(h.m,{},'player');
 assert.equal(h.calls.length,0);assert.equal(h.errors.length,0);assert.deepEqual(h.m.flags,before);h.unregister();
});
for(const proof of ['local','native'])test(`opted-in ${proof} actual Use dispatches exactly once`,async()=>{
 const h=harness({requiresActualUse:()=>true});
 if(proof==='local')h.m.flags[MODULE_ID]={usageInput:{actualUse:true}};
 else h.m.flags.pf2e.origin.rollOptions=['origin:action:slug:use-action'];
 await Promise.all([h.callbacks.get('createChatMessage')(h.m,{},'player'),h.callbacks.get('createChatMessage')(h.m,{},'player')]);
 assert.equal(h.calls.length,1);assert.equal(h.m.flags[MODULE_ID].usage.status,'done');assert.equal(h.errors.length,0);h.unregister();
});
test('a policy that does not select the activity preserves legacy display-card dispatch',async()=>{
 const h=harness({requiresActualUse:(_item,action)=>action==='spell-combination:combination'});
 await h.callbacks.get('createChatMessage')(h.m,{},'player');assert.equal(h.calls.length,1);assert.equal(h.m.flags[MODULE_ID].usage.status,'done');h.unregister();
});
