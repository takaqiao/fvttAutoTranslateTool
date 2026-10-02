import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {MODULE_ID} from '../scripts/rules.mjs';
import {createThrallAutomation,CONSUME_THRALL_SOURCE} from '../scripts/thrall-automation.mjs';

const documentSource=process.env.FOUNDRY_DOCUMENT_SOURCE;
assert.ok(documentSource,'FOUNDRY_DOCUMENT_SOURCE must identify the pinned Foundry 14.368 abstract Document source');
const hash=value=>createHash('sha256').update(value).digest('hex'),bytes=readFileSync(documentSource),source=bytes.toString('utf8');
assert.equal(hash(bytes),'303e6bc84fbaa7d564dab144b1f59fd0a1f6584a14e872b293e74358ed9b49dc');
const start=source.indexOf('  testUserPermission(user, permission, {exact=false}={}) {'),end=source.indexOf('\n  }',start)+4;
assert.ok(start>=0&&end>start);
const method=source.slice(start,end);
assert.equal(hash(method),'01701d706609317e7cf476bfb4abdc7ecfac8effc9001a9fd0df1b86761a80de');
const nativePermission=Function('DOCUMENT_OWNERSHIP_LEVELS',`return ({${method}}).testUserPermission;`)({NONE:0,LIMITED:1,OBSERVER:2,OWNER:3});
function patch(doc,changes){for(const[key,value]of Object.entries(changes)){const parts=key.split('.');let at=doc;for(const part of parts.slice(0,-1))at=at[part]??={};at[parts.at(-1)]=structuredClone(value);}}

function fixture({replaceAt,replacement={isGM:false,isBanned:true}}={}){
 const gm={id:'gm',isGM:true,active:true},user={id:'owner',isGM:true,active:true},users=Object.assign(new Map([[gm.id,gm],[user.id,user]]),{activeGM:gm});
 const game={user:gm,users,actors:new Map(),scenes:new Map(),messages:new Map(),time:{worldTime:10}},scene={id:'scene',tokens:new Map()},docs=new Map(),counts={deletes:0,grants:0,payments:0};
 const actor={id:'summoner',uuid:'Actor.summoner',type:'character',canAct:true,items:new Map(),flags:{},system:{resources:{focus:{value:0,max:1}}},getUserLevel:()=>0,testUserPermission:nativePermission,async update(changes){patch(this,changes);if(changes['system.resources.focus.value']===1)counts.grants++;}};
 const item={id:'consume',uuid:actor.uuid+'.Item.consume',actor,sourceId:CONSUME_THRALL_SOURCE,system:{frequency:{value:1,max:1,per:'day'}},async update(changes){counts.payments++;patch(this,changes)}};actor.items.set(item.id,item);
 const origin={id:'source',uuid:'Scene.scene.Token.source',documentName:'Token',parent:scene,actor,object:{distanceTo:()=>5}},targetActor={uuid:'Actor.thrall',flags:{'pf2e-summons-assistant':{summoner:{uuid:actor.uuid}}},rollOptions:{all:{'self:trait:thrall':true}},system:{attributes:{hp:{value:1}}}};
 const target={id:'thrall',uuid:'Scene.scene.Token.thrall',documentName:'Token',parent:scene,actor:targetActor,object:{},async delete(){counts.deletes++;docs.delete(this.uuid);scene.tokens.delete(this.id);throw Error('native deletion reply lost')}};
 for(const token of [origin,target]){scene.tokens.set(token.id,token);docs.set(token.uuid,token);}game.actors.set(actor.id,actor);game.scenes.set(scene.id,scene);
 const message={id:'use',author:user,speaker:{actor:actor.id,scene:scene.id,token:origin.id},flags:{pf2e:{origin:{actor:actor.uuid,uuid:item.uuid}},[MODULE_ID]:{usageInput:{actualUse:true,targetUuids:[target.uuid]}}}};game.messages.set(message.id,message);
 let reconcile=false,replaced=false;
 const provider=createThrallAutomation({game,castEvents:{addActorMatcher(){},addConsumePolicy(){}},fromUuid:async uuid=>{
  if(reconcile&&!replaced&&uuid===(replaceAt==='source'?origin.uuid:replaceAt==='target'?target.uuid:null)){replaced=true;users.set(user.id,{...user,...replacement});}
  return docs.get(uuid);
 }});
 return {game,user,actor,item,provider,counts,async enterUnknown(){await assert.rejects(provider.executeUsage({actor,item,message,user,action:'thrall:consume'}),/reply lost/);assert.equal(actor.flags[MODULE_ID].thrallUse.status,'uncertain');reconcile=true;}};
}

test('native missing-target reconciliation grants once with the original User document',async()=>{
 const f=fixture();await f.enterUnknown();await f.provider.maintain(f.actor);await f.provider.maintain(f.actor);
 assert.equal(f.actor.system.resources.focus.value,1);assert.equal(f.actor.flags[MODULE_ID].thrallUse.status,'complete');assert.equal(f.item.system.frequency.value,0);assert.deepEqual(f.counts,{deletes:1,grants:1,payments:1});
});
for(const replaceAt of ['source','target'])test(`same-ID owner losing native OWNER during ${replaceAt} lookup retains uncertain reconciliation`,async()=>{
 const f=fixture({replaceAt});await f.enterUnknown();await f.provider.maintain(f.actor);
 assert.equal(f.actor.testUserPermission(f.user,'OWNER'),true);assert.equal(f.actor.testUserPermission(f.game.users.get(f.user.id),'OWNER'),false);
 assert.equal(f.actor.system.resources.focus.value,0);assert.equal(f.actor.flags[MODULE_ID].thrallUse.status,'uncertain');assert.equal(f.item.system.frequency.value,0);assert.deepEqual(f.counts,{deletes:1,grants:0,payments:1});
});
test('same-ID owner replacement preserves uncertainty even when its native OWNER level is unchanged',async()=>{
 const f=fixture({replaceAt:'target',replacement:{}});await f.enterUnknown();await f.provider.maintain(f.actor);
 assert.equal(f.actor.testUserPermission(f.game.users.get(f.user.id),'OWNER'),true);assert.notEqual(f.game.users.get(f.user.id),f.user);
 assert.equal(f.actor.system.resources.focus.value,0);assert.equal(f.actor.flags[MODULE_ID].thrallUse.status,'uncertain');assert.equal(f.item.system.frequency.value,0);assert.deepEqual(f.counts,{deletes:1,grants:0,payments:1});
});
