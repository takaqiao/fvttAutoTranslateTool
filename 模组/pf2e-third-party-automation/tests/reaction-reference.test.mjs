import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHalflingLuckProvider} from '../scripts/halfling-luck.mjs';
import {createHalflingLuckLedger,HALFLING_LUCK_SOURCE} from '../scripts/halfling-luck-ledger.mjs';
import {runCheckReactionPipeline} from '../scripts/reaction-checks.mjs';

const ID='pf2e-third-party-automation',copy=value=>structuredClone(value);
function apply(target,changes){for(const [key,value]of Object.entries(changes)){const parts=key.split('.');let at=target;for(const part of parts.slice(0,-1))at=at[part]??={};at[parts.at(-1)]=copy(value)}}
function fixture(){
 const gm={id:'gm',isGM:true,active:true},owner={id:'owner',isGM:false,active:true},offlineGM={id:'offline-gm',isGM:true,active:false};
 const users=new Map([[gm.id,gm],[owner.id,owner],[offlineGM.id,offlineGM]]);users.activeGM=gm;
 const actor={id:'actor',uuid:'Actor.actor',type:'character',canAct:true,isDead:false,flags:{},items:new Map(),testUserPermission:user=>user===owner||user.isGM};
 const item={id:'luck',uuid:'Actor.actor.Item.luck',type:'feat',sourceId:HALFLING_LUCK_SOURCE,actor,flags:{},_source:{system:{frequency:{max:1,per:'day'}}},system:{actionType:{value:'free'},frequency:{value:1,max:1,per:'day'}},async update(changes){apply(item,changes);return item}};
 actor.items.set(item.id,item);
 const messages=new Map(),docs=new Map([[actor.uuid,actor],[item.uuid,item]]),game={user:gm,users,actors:new Map([[actor.id,actor]]),messages,time:{worldTime:100},pf2e:{}};let sequence=0;
 const ledger=createHalflingLuckLedger({game,fromUuid:async uuid=>docs.get(uuid),randomId:()=>`nonce-${++sequence}`});
 const original={total:8,options:{degreeOfSuccess:1},toJSON(){return {class:'CheckRoll',evaluated:true,total:8,options:{degreeOfSuccess:1},terms:[{class:'Die',number:1,faces:20,results:[{result:3,active:true}]}]}},async render(){return '<div class="dice-roll">原骰 3 + 5 = 8</div>'}};
 const nativeData={user:gm.id,blind:true,whisper:[gm.id],speaker:{actor:actor.id},flavor:'private native flavor',rolls:[original.toJSON()],flags:{pf2e:{context:{type:'saving-throw',outcome:'failure',messageMode:'blind',options:[]}},other:{nativeData:'preserve'}}};
 const references=[],finalPublications=[],callbacks=[],nativeCalls=[];
 const publishReference=async data=>{const message={id:`reference-${++sequence}`,uuid:`ChatMessage.reference-${sequence}`,...copy(data)};references.push(message);messages.set(message.id,message);docs.set(message.uuid,message);return message};
 let provider;
 const makeProvider=(referencePublisher)=>createHalflingLuckProvider({game,ledger,fromUuid:async uuid=>docs.get(uuid),assess:()=>({eligible:true}),choose:async()=> 'use',randomId:()=> 'native-invocation',referencePublisher,
  publish:async data=>{finalPublications.push(data);return {id:'final-card',...data}},
  originalUse:async used=>{
   const changes={'system.frequency.value':0},options={};assert.equal(ledger.preparePayment(used,changes,options,gm.id),true);
   const receipt={id:'native-frequency',itemUuid:used.uuid,userId:gm.id,before:1,after:0,createdAt:100};options[ID]={...options[ID],frequencyReceipt:receipt};
   await used.update(changes);ledger.observePayment(used,changes,options,gm.id);
   const message={id:'native-use',uuid:'ChatMessage.native-use',author:gm,speaker:{actor:actor.id},rolls:[],flags:{pf2e:{origin:{uuid:used.uuid,actor:actor.uuid,type:'feat'}},[ID]:{...provider.captureUsage(used),usageInput:{actualUse:true,frequencyReceiptId:receipt.id}}}};
   messages.set(message.id,message);docs.set(message.uuid,message);await provider.executeUsage({actor,item:used,user:gm,message,frequencyReceipt:receipt});
  }
 });
 function install(referencePublisher=publishReference){provider=makeProvider(referencePublisher);provider.register({Hooks:{on:()=>1,off(){}}});return provider}
 const native=async(check,context,event,callback)=>{nativeCalls.push({check,context,event});if(nativeCalls.length===2)return null;await callback(original,'failure',{toObject:()=>copy(nativeData),flags:nativeData.flags},event);return original};
 const context={actor,type:'saving-throw',domains:['will'],options:new Set(),dc:{value:18},createMessage:true};
 return {game,gm,owner,offlineGM,actor,item,ledger,messages,docs,original,nativeData,references,finalPublications,callbacks,nativeCalls,context,native,publishReference,install,run:(nativeRoll=native)=>provider.interceptCheck(nativeRoll,{slug:'will',modifiers:[]},context,null,(...args)=>callbacks.push(args))};
}

for(const createMessage of [true,false])for(const failure of ['cancel','throw'])test(`actual Halfling provider + ledger retain a GM-only original reference on paid ${failure}; createMessage=${createMessage}`,async()=>{
 const f=fixture();f.install();f.context.createMessage=createMessage;const originalError=Error('second native disconnected');
 const native=async(...args)=>{if(f.nativeCalls.length===1&&failure==='throw'){f.nativeCalls.push({});throw originalError}return f.native(...args)};
 await assert.rejects(f.run(native),error=>{if(failure==='throw')assert.equal(error,originalError);else assert.match(error.message,/半身人幸运.*重掷未完成/);return true});
 assert.equal(f.ledger.current(f.item).status,'uncertain');assert.equal(f.item.system.frequency.value,0);assert.equal(f.nativeCalls.length,2);assert.equal(f.finalPublications.length,0);assert.equal(f.callbacks.length,0);
 assert.equal(f.references.length,1,'paid original die must survive the rejected pipeline');
 const card=f.references[0],proof=card.flags[ID].reference;
 assert.equal(f.messages.get(card.id),card);assert.equal(card.blind,true);assert.deepEqual(card.whisper,['gm','offline-gm']);assert.deepEqual(card.rolls,[]);
 assert.equal(card.flags.pf2e,undefined);assert.equal(card.flags[ID].reactionChecks,undefined);assert.match(card.content,/原检定参考[，,]?\s*非最终结果/);
 assert.deepEqual(proof.rollJSON,f.original.toJSON());assert.deepEqual(proof.nativeCardData,f.nativeData);assert.equal(proof.nonce,f.ledger.current(f.item).nonce);assert.equal(proof.actorUuid,f.actor.uuid);assert.equal(proof.reaction,'halfling-luck');
 await assert.rejects(f.ledger.startRolling({actor:f.actor,item:f.item,user:f.gm,nonce:proof.nonce}));assert.equal(f.references.length,1);
});

function clockFixture(){
 const f=fixture(),nonce='paid-clock';f.actor.flags[ID]={reactionChecks:{reactions:[{nonce,kind:'clock',state:'claimed',userId:f.gm.id,checkId:'clock-use'}]}};
 f.messages.set('clock-use',{id:'clock-use',speaker:{actor:f.actor.id},flags:{[ID]:{reactionChecks:{kind:'reaction-use',reaction:'clock',nonce}}}});
 f.game.pf2e.Modifier=class{constructor(data){Object.assign(this,data)}};f.game.pf2e.CheckModifier=class{constructor(slug,base,extra){this.slug=slug;this.modifiers=[...base.modifiers,...extra]}};
 const args={game:f.game,check:{slug:'will',modifiers:[]},context:f.context,native:f.native,decide:async()=>({reaction:'clock',nonce,actorUuid:f.actor.uuid}),publish:async data=>{f.finalPublications.push(data)},referencePublisher:f.publishReference,callback:(...args)=>f.callbacks.push(args)};
 return {...f,args,run:()=>runCheckReactionPipeline(args)};
}

test('paid Clock reference remains readable by all GMs after primary GM handoff',async()=>{
 const f=clockFixture(),native=f.args.native;f.args.native=async(...args)=>{if(f.nativeCalls.length===1)f.game.users.activeGM=f.offlineGM;return native(...args)};
 await assert.rejects(f.run(),/倒转光阴.*重掷未完成/);assert.equal(f.references.length,1);assert.deepEqual(f.references[0].whisper,['gm','offline-gm']);assert.equal(f.finalPublications.length,0);assert.equal(f.callbacks.length,0);
});

for(const failure of ['no-gm','no-publisher','publisher-cancel','publisher-throw','unverified-nonce','foreign-actor'])test(`paid reroll ${failure} cannot silently claim a saved reference or publish publicly`,async()=>{
 const f=clockFixture();
 if(failure==='no-gm'){f.gm.isGM=false;f.offlineGM.isGM=false}
 if(failure==='no-publisher')f.args.referencePublisher=null;
 if(failure==='publisher-cancel')f.args.referencePublisher=async()=>null;
 if(failure==='publisher-throw')f.args.referencePublisher=async()=>{throw Error('storage refused')};
 if(failure==='unverified-nonce')f.args.decide=async()=>({reaction:'clock',nonce:'unknown',actorUuid:f.actor.uuid});
 if(failure==='foreign-actor')f.args.decide=async()=>({reaction:'clock',nonce:'paid-clock',actorUuid:'Actor.foreign'});
 await assert.rejects(f.run(),error=>{assert.match(error.message,/重掷未完成/);assert.match(error.message,/原检定参考未保存/);return true});
 assert.equal(f.references.length,0);assert.equal(f.finalPublications.length,0);assert.equal(f.callbacks.length,0);assert.equal(f.nativeCalls.length,2);
});

test('successful and initially cancelled native checks never create reference-only cards',async()=>{
 const f=clockFixture();f.args.decide=async()=>null;f.game.pf2e.Check={};f.args.publish=async data=>({id:'native-final',...data});
 assert.equal(await f.run(),f.original);assert.equal(f.references.length,0);assert.equal(f.callbacks.length,1);
 const other=clockFixture();other.args.native=async()=>null;assert.equal(await other.run(),null);assert.equal(other.references.length,0);assert.equal(other.callbacks.length,0);
});
