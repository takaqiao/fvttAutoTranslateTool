import test from 'node:test';
import assert from 'node:assert/strict';
import {createElementalMedicine} from '../scripts/elemental-medicine.mjs';
import {ELEMENTAL_MEDICINE_SOURCE,ELEMENTAL_MEDICINE_DAILY,ELEMENTAL_MEDICINE_EFFECT} from '../scripts/elemental-medicine-rules.mjs';
import {MODULE_ID as ID} from '../scripts/rules.mjs';

function set(doc,changes){for(const[path,value]of Object.entries(changes)){const parts=path.split('.');let cursor=doc;for(const key of parts.slice(0,-1))cursor=cursor[key]??={};cursor[parts.at(-1)]=structuredClone(value);}}
function setup({degree=null,lostReply=false,pending=false}={}){
 const gm={id:'gm',isGM:true,active:true},player={id:'player',active:true},users=Object.assign(new Map([[gm.id,gm],[player.id,player]]),{activeGM:gm});
 const callbacks=new Map();let next=0;const Hooks={on(name,callback){const id=++next;callbacks.set(id,{name,callback});return id;},off(_name,id){callbacks.delete(id);},call(name,...args){for(const h of [...callbacks.values()])if(h.name===name)h.callback(...args);}};
 const binding={itemUuid:'Actor.patient.Item.affliction',itemId:'affliction',originUuid:'Actor.patient',originSignature:'patient-signature',predicate:['item:id:affliction','origin:signature:patient-signature']};
 const facts={id:'facts',author:gm,blind:true,whisper:[gm.id],flags:{[ID]:{elementalMedicineFacts:{requestUuid:'Actor.doctor.Item.request',facts:[{patientUuid:'Actor.patient',skill:'medicine',dc:15,binding,correctElement:'wood',wrongElement:'earth'}]}}}};
 const messages=new Map([[facts.id,facts]]),counts={gm:0,player:0},notices=[],dialogs=[];
 const realms=[gm,player].map(user=>{
  const game={user,users,messages,time:{worldTime:100},actors:new Map(),modules:new Map(),scenes:new Map()};
  const doctor={id:'doctor',uuid:'Actor.doctor',type:'character',items:new Map(),flags:{},testUserPermission:()=>true,async update(changes){set(this,changes);},getStatistic:()=>({check:{async roll(parameters){
   counts[user.id]++;assert.equal(parameters.skipDialog,false);assert.equal(parameters.messageMode,'blind');assert.equal(parameters.dc.visible,false);assert.equal(parameters.dc.value,15);
   if(pending)return new Promise(resolve=>{const dialog={context:{options:new Set(parameters.extraRollOptions)},resolve,async close(){}};dialogs.push(dialog);Hooks.call('renderCheckModifiersDialog',dialog);});
   if(degree===null)return null;
   const roll={_evaluated:true,total:20,options:{degreeOfSuccess:degree}},card={id:'native-check',author:user,actor:doctor,speaker:{actor:doctor.id},blind:false,whisper:[],rolls:[roll],flags:{pf2e:{context:{type:'skill-check',dc:parameters.dc,options:[...parameters.extraRollOptions,'check:statistic:medicine'],outcome:['criticalFailure','failure','success','criticalSuccess'][degree]}}},updateSource(changes){set(this,changes);}};
   Hooks.call('preCreateChatMessage',card,{}, {},user.id);messages.set(card.id,card);Hooks.call('createChatMessage',card);return roll;
  }}})};
  doctor.items.set('feat',{id:'feat',uuid:'Actor.doctor.Item.feat',actor:doctor,type:'feat',sourceId:ELEMENTAL_MEDICINE_SOURCE});
  const request={id:'request',uuid:'Actor.doctor.Item.request',actor:doctor,flags:{'pf2e-dailies':{daily:`module.${ELEMENTAL_MEDICINE_DAILY}`},[ID]:{elementalMedicine:{kind:'preparation',actorUuid:doctor.uuid,userId:player.id,status:'diagnosing',factsId:facts.id,patients:[{patientUuid:'Actor.patient',skill:'medicine',state:'pending'}]}}},async update(changes){for(const realm of realms)set(realm.request,changes);for(const h of callbacks.values())if(h.name==='updateItem')h.callback(this,changes);}};
  doctor.items.set(request.id,request);game.actors.set(doctor.id,doctor);
  const patient={uuid:'Actor.patient',signature:'patient-signature',type:'character',level:1,testUserPermission:()=>true,items:new Map(),async createEmbeddedDocuments(_kind,rows){return rows.map(data=>{const doc={...data,id:data._id,sourceId:ELEMENTAL_MEDICINE_EFFECT};this.items.set(doc.id,doc);return doc;});}};
  const affliction={id:'affliction',uuid:binding.itemUuid,actor:patient,getRollOptions:()=>['item:id:affliction']};patient.items.set(affliction.id,affliction);
  const fromUuid=async uuid=>uuid===request.uuid?request:uuid===patient.uuid?patient:uuid===doctor.uuid?doctor:uuid===affliction.uuid?affliction:uuid===ELEMENTAL_MEDICINE_EFFECT?{toObject:()=>({type:'effect',system:{rules:[]}})}:null;
  const endpoints=new Map(),socket={register(name,fn){endpoints.set(name,fn);},executeAsUser(name,userId,payload){const target=realms.find(r=>r.game.user.id===userId),result=target.endpoints.get(name).call({socketdata:{userId:user.id}},payload);return lostReply?Promise.resolve(result).then(()=>new Promise(()=>{})):result;}};
  const provider=createElementalMedicine({game,fromUuid,publishDiagnosis:async context=>{notices.push({client:user.id,...context});return {id:'notice'};}});
  return {game,request,patient,endpoints,socket,provider};
 });
 for(const realm of realms)realm.provider.register({Hooks,socket:realm.socket});
 return {gm,player,realms,counts,Hooks,notices,dialogs};
}

test('Dailies diagnosis opens only the preparation author native check and retains an unrolled cancellation',async()=>{
 const {player,realms,counts}=setup();
 await assert.rejects(()=>realms[0].provider.prepare(realms[0].request.uuid,player),/回执|取消/);
 assert.deepEqual(counts,{gm:0,player:1});
 const row=realms[0].request.flags[ID].elementalMedicine.patients[0];assert.equal(row.rollUserId,player.id);
 assert.equal(JSON.stringify(realms[0].request.flags).includes('"dc"'),false);
 await assert.rejects(()=>realms[0].provider.prepare(realms[0].request.uuid,player),/回执|取消/);assert.deepEqual(counts,{gm:0,player:1});
});

for(const lostReply of [false,true])test(`Dailies owner blind result settles medicine once${lostReply?' even when its socket reply is lost':''}`,{timeout:1000},async()=>{
 const {gm,player,realms,counts,notices}=setup({degree:2,lostReply});
 assert.equal((await realms[0].provider.prepare(realms[0].request.uuid,player)).status,'done');
 const card=realms[0].game.messages.get('native-check');assert.equal(card.author,player);assert.equal(card.blind,true);assert.deepEqual(card.whisper,[gm.id]);
 assert.equal(notices.length,1);assert.equal(notices[0].client,gm.id);assert.ok(notices[0].recipients.includes(player.id));
 assert.equal([...realms[0].patient.items.values()].filter(i=>i.flags?.[ID]?.elementalMedicine?.kind==='medicine').length,1);
 await realms[0].provider.prepare(realms[0].request.uuid,player);assert.deepEqual(counts,{gm:0,player:1});assert.equal(notices.length,1);
 assert.equal(JSON.stringify(realms[0].request.flags).includes('"dc"'),false);
});

for(const change of ['disconnect','handoff'])test(`Dailies pending owner window stops on ${change} without diagnosis`,{timeout:1000},async()=>{
 const f=setup({pending:true}),started=f.realms[0].provider.prepare(f.realms[0].request.uuid,f.player);
 const rejected=assert.rejects(started,/改变|失效|无效/);
 for(let i=0;i<10&&!f.dialogs.length;i++)await new Promise(resolve=>setImmediate(resolve));assert.equal(f.dialogs.length,1);
 if(change==='disconnect'){f.player.active=false;f.Hooks.call('userConnected',f.player,false);}else{f.realms[0].game.users.activeGM={id:'new-gm',isGM:true,active:true};f.Hooks.call('updateUser',f.gm);}
 await rejected;assert.equal(f.realms[0].game.messages.has('native-check'),false);assert.equal(f.notices.length,0);assert.deepEqual(f.counts,{gm:0,player:1});
});
