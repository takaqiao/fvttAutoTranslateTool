import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createNativeTargetSaves,selectNativeTargetSaveOwner} from '../scripts/native-target-saves.mjs';

const ID='pf2e-third-party-automation',outcomes=['criticalFailure','failure','success','criticalSuccess'];
function patch(doc,changes){for(const[path,value]of Object.entries(changes)){const keys=path.split('.');let node=doc;for(const key of keys.slice(0,-1))node=node[key]??={};node[keys.at(-1)]=structuredClone(value);}}
function hooks(){const entries=new Map();let seq=0;return {entries,on(event,fn){const id=++seq;entries.set(id,{event,fn});return id},off(_event,id){entries.delete(id)},call(event,...args){let allowed=true;for(const entry of [...entries.values()])if(entry.event===event&&entry.fn(...args)===false)allowed=false;return allowed}};}

function fixture({scope='spell-combination',cancel=false,hold=false,lostReply=false,blind=false,mode='public',wrongCard,lagTerminal=false}={}){
 const gm={id:'gm',isGM:true,active:true},caster={id:'caster',active:true},player={id:'target-player',active:true,character:'Actor.target'},otherGM={id:'next-gm',isGM:true,active:true};
 const users=new Map([gm,caster,player,otherGM].map(user=>[user.id,user]));users.activeGM=gm;
 const realms=[],calls={native:[],dice:0,updates:[],variants:[],rpc:[],rpcErrors:[]},delayed=[];let accept,opened,releaseTerminal,ownerFinished;
 const windowOpened=new Promise(resolve=>opened=resolve),terminalGate=new Promise(resolve=>releaseTerminal=resolve),ownerSettled=new Promise(resolve=>ownerFinished=resolve);
 function realm(user){
  const Hooks=hooks(),messages=new Map(),actors=new Map(),handlers=new Map(),docs=new Map();
  const game={user,users,messages,actors};
  const sourceActor={id:'caster-actor',uuid:'Actor.caster-actor',type:'character',items:new Map(),testUserPermission:u=>u.id==='caster'};
  const targetActor={id:'target',uuid:'Actor.target',type:'character',items:new Map(),testUserPermission:u=>u.id==='target-player'};
  actors.set(sourceActor.id,sourceActor);actors.set(targetActor.id,targetActor);
  const scene={id:'scene',tokens:new Map()},target={id:'target-token',uuid:'Scene.scene.Token.target-token',actor:targetActor,parent:scene};scene.tokens.set(target.id,target);
  const item={id:'spell',uuid:sourceActor.uuid+'.Item.spell',actor:sourceActor,type:scope==='spell-combination'?'spell':'action',name:'Source',rank:1,system:{traits:{value:['incapacitation']}},loadVariant(parameters){calls.variants.push({user:user.id,...parameters});return {...this,rank:parameters.castRank,appliedOverlays:new Map(parameters.overlayIds.map(id=>[id,id]))};}};sourceActor.items.set(item.id,item);
  const source={id:'activity',author:caster,blind,whisper:blind?['gm']:[],flags:{pf2e:{origin:{actor:sourceActor.uuid,uuid:item.uuid,castRank:8},context:{type:'damage-taken'}},[ID]:scope==='spell-combination'?{usageGenerated:true,spellCombination:{activityMessageId:'original-cast'}}:{spiritualScarFollowup:{status:'rolling',nonce:'scar-claim',actorUuid:sourceActor.uuid,itemUuid:item.uuid,targetActorUuid:targetActor.uuid,targetTokenUuid:target.uuid,damageMessageId:'activity'}}},async update(changes){calls.updates.push({user:user.id,changes:structuredClone(changes)});for(const r of realms){const apply=()=>{patch(r.source,changes);r.Hooks.call('updateChatMessage',r.source)};if(lagTerminal&&r.game.user.id==='target-player'&&Object.values(changes).some(value=>['done','uncertain'].includes(value?.status)))delayed.push(apply);else apply()}return this;}};
  messages.set(source.id,source);for(const doc of [sourceActor,item,target])docs.set(doc.uuid,doc);
  const r={game,Hooks,handlers,docs,sourceActor,targetActor,target,item,source};realms.push(r);
  targetActor.getStatistic=statistic=>({check:{async roll(parameters){
   calls.native.push({user:user.id,parameters});assert.equal(parameters.skipDialog,false);assert.equal(parameters.event,null);assert.equal(parameters.createMessage,true);
   let resolve;const accepted=new Promise(done=>resolve=done),app={context:{options:new Set(parameters.extraRollOptions)},resolve,close(){resolve(false)}};
   Hooks.call('renderCheckModifiersDialog',app);accept=value=>app.resolve(value);opened();if(!hold)await app.resolve(!cancel);
   if(!await accepted)return null;calls.dice++;
   const degree=2,roll={_evaluated:true,total:32,options:{degreeOfSuccess:degree}};
   const card={id:'native-save',author:user,speaker:{actor:targetActor.id,scene:scene.id,token:target.id},actor:targetActor,rolls:[roll],blind:mode==='blind',whisper:mode==='public'?[]:mode==='self'?[user.id]:['gm'],flags:{pf2e:{origin:{actor:parameters.origin.uuid,uuid:parameters.item.uuid,castRank:parameters.item.rank,variant:{overlays:[...parameters.item.appliedOverlays?.values?.()??[]]}},context:{type:'saving-throw',action:parameters.action,options:[...parameters.extraRollOptions,'check:statistic:'+statistic],dc:structuredClone(parameters.dc),outcome:outcomes[degree],messageMode:mode}}},updateSource(changes){patch(this,changes)}};
   wrongCard?.(card);if(Hooks.call('preCreateChatMessage',card,{}, {},user.id)===false)return null;
   for(const peer of realms){peer.game.messages.set(card.id,card);peer.Hooks.call('createChatMessage',card)}
   await parameters.callback(roll,outcomes[degree],card);return roll;
  }}});
  r.socket={register(name,handler){handlers.set(name,handler)},async executeAsUser(name,id,payload){calls.rpc.push({user:user.id,id,name,payload:structuredClone(payload)});const destination=realms.find(peer=>peer.game.user.id===id);assert.ok(destination,'destination user realm exists');if(lagTerminal&&name.endsWith(':stage')&&payload.status==='done')await terminalGate;try{const result=await destination.handlers.get(name).call({socketdata:{userId:user.id}},payload);if(lostReply&&name===`native-target-save:${scope}`)return new Promise(()=>{});return result;}catch(error){calls.rpcErrors.push({user:user.id,status:payload.status,error});throw error;}finally{if(name===`native-target-save:${scope}`)ownerFinished();}}};
  r.service=createNativeTargetSaves({game,fromUuid:async uuid=>docs.get(uuid),scope,syncTimeoutMs:10});r.service.register({Hooks,socket:r.socket});return r;
 }
 const root=realm(gm),owner=realm(player);
 const context={sourceActor:root.sourceActor,sourceItem:root.item,sourceMessage:root.source,target:root.target};
 const request={statistic:scope==='spell-combination'?'reflex':'will',action:scope==='spell-combination'?'spell-combination-save':'spiritual-scar',dc:{value:29},traits:['incapacitation'],options:['secret:rule-option'],rank:8,overlayIds:['overlay'],minimumPrivacy:{blind,whisper:blind?['gm']:[]}};
 return {root,owner,realms,realm,calls,context,request,gm,otherGM,caster,player,users,windowOpened,ownerSettled,releaseTerminal,flushTerminal:()=>{for(const apply of delayed.splice(0))apply()},accept:value=>accept(value),run:()=>root.service.run(context,request),state:()=>Object.values(root.source.flags[ID].nativeTargetSaves??{})[0],remainingHooks:()=>realms.reduce((sum,r)=>sum+r.Hooks.entries.size,0)};
}
async function setup(parameters,fn){const previous=globalThis.foundry;let nonce=0;globalThis.foundry={utils:{randomID:()=>`target-save-${++nonce}`}};try{await fn(fixture(parameters));}finally{globalThis.foundry=previous;}}
const settled=()=>new Promise(resolve=>setImmediate(resolve));
const lateStage=(f,changes={},user=f.player)=>f.root.handlers.get('native-target-save:'+f.state().scope+':stage').call({socketdata:{userId:user.id}},{messageId:f.root.source.id,nonce:f.state().binding.nonce,status:'done',messageIdResult:f.state().messageId,...changes});

test('a foreign-source PC save runs once on the bound target player and keeps the caster variant and native card',()=>setup({},async f=>{
 const result=await f.run();await settled();assert.equal(result.status,'rolled');assert.equal(result.messageId,'native-save');assert.equal(result.check.author.id,f.player.id);assert.equal(f.calls.dice,1);assert.equal(f.calls.native[0].user,f.player.id);assert.equal(f.calls.native[0].parameters.origin.uuid,f.context.sourceActor.uuid);assert.notEqual(f.calls.native[0].parameters.item,f.owner.item,'a rank variant is not the live base item');assert.equal(f.calls.native[0].parameters.item.rank,8);assert.deepEqual(f.calls.variants,[{user:f.player.id,castRank:8,overlayIds:['overlay']}]);assert.equal(f.state().status,'done');assert.equal(f.state().messageId,'native-save');assert.equal(f.remainingHooks(),0);
 assert.equal(f.root.source.author.id,f.caster.id);assert.ok(f.calls.updates.every(update=>update.user==='gm'));
 const publicOperation=JSON.stringify(f.state());for(const secret of ['29','incapacitation','secret:rule-option','"dc"','"roll"','"total"'])assert.equal(publicOperation.includes(secret),false,secret+' is not in the source operation');
}));
test('closing the target player window produces no native result or replacement GM roll',()=>setup({cancel:true},async f=>{const result=await f.run();await settled();assert.deepEqual(result,{status:'cancelled'});assert.equal(f.calls.dice,0);assert.equal(f.root.game.messages.size,1);assert.equal(f.state().cancelled,true);assert.equal(f.remainingHooks(),0)}));
test('re-entering the same saved target save never opens a second window or rolls again',()=>setup({},async f=>{const first=await f.run();const second=await f.run();await settled();assert.equal(second.messageId,first.messageId);assert.equal(f.calls.native.length,1);assert.equal(f.calls.dice,1);assert.equal(f.remainingHooks(),0)}));
test('a blind source selects native blind mode before reactions or publication',()=>setup({blind:true},async f=>{await f.run();assert.equal(f.calls.native[0].parameters.messageMode,'blind')}));
test('a matching save that loses its unique marker cannot publish an unproved default-public card',()=>setup({blind:true,wrongCard:card=>{card.flags.pf2e.context.options=card.flags.pf2e.context.options.filter(option=>!option.includes(':target-save:'));}},async f=>{await assert.rejects(f.run());await settled();assert.equal(f.root.game.messages.has('native-save'),false);assert.equal(f.remainingHooks(),0)}));
test('changing the public operation binding while the player window is open aborts before any die',()=>setup({hold:true},async f=>{const operation=f.run();await f.windowOpened;const nonce=f.state().binding.nonce;await f.root.source.update({[`flags.${ID}.nativeTargetSaves.${nonce}`]:{...f.state(),binding:{...f.state().binding,sourceItemUuid:'Actor.other.Item.other'}}});await assert.rejects(operation,/改变/);await settled();assert.equal(f.calls.dice,0);assert.equal(f.remainingHooks(),0)}));
test('an offline target player fails before creating a public operation or opening any GM window',()=>setup({},async f=>{f.player.active=false;await assert.rejects(f.run(),/不在线/);assert.equal(f.calls.native.length,0);assert.equal(f.calls.updates.length,0)}));
test('the target player can leave a window open beyond the document-sync timeout',()=>setup({hold:true},async f=>{const operation=f.run();await f.windowOpened;await new Promise(resolve=>setTimeout(resolve,30));assert.equal(f.calls.dice,0);await f.accept(true);assert.equal((await operation).status,'rolled');await settled();assert.equal(f.calls.dice,1);assert.equal(f.remainingHooks(),0)}));
test('a saved exact native result wakes the GM when the owner socket reply is lost',()=>setup({lostReply:true},async f=>{let timer;try{const result=await Promise.race([f.run(),new Promise((_,reject)=>timer=setTimeout(()=>reject(Error('lost reply still blocks durable result')),200))]);assert.equal(result.messageId,'native-save');await settled();assert.equal(f.calls.dice,1);assert.equal(f.remainingHooks(),0)}finally{clearTimeout(timer)}}));
test('the owner done RPC remains valid after the GM create-card observer finishes and the private request is cleaned',()=>setup({lagTerminal:true},async f=>{
 const result=await f.run();assert.equal(result.messageId,'native-save');assert.equal(f.state().status,'done');assert.equal(Object.values(f.owner.source.flags[ID].nativeTargetSaves)[0].status,'started','the owner has not received the GM terminal update');
 f.releaseTerminal();await f.ownerSettled;f.flushTerminal();await settled();assert.deepEqual(f.calls.rpcErrors,[]);assert.equal(f.calls.dice,1);assert.equal(f.remainingHooks(),0);
}));
test('a late uncertain notification cannot overwrite the already verified native result',()=>setup({},async f=>{await f.run();await f.ownerSettled;const before=structuredClone(f.state()),writes=f.calls.updates.length;assert.deepEqual(await lateStage(f,{status:'uncertain',messageIdResult:undefined}),{status:'rolled',messageId:'native-save'});assert.deepEqual(f.state(),before);assert.equal(f.calls.updates.length,writes);assert.equal(f.calls.dice,1)}));
test('a late Spiritual Scar done acknowledgement survives the followup advancing beyond rolling',()=>setup({scope:'spiritual-scar'},async f=>{await f.run();await f.ownerSettled;await f.root.source.update({[`flags.${ID}.spiritualScarFollowup.status`]:'done'});assert.deepEqual(await lateStage(f),{status:'rolled',messageId:'native-save'});assert.deepEqual(f.calls.rpcErrors,[]);assert.equal(f.calls.dice,1);assert.equal(f.remainingHooks(),0)}));
test('a saved cancellation accepts only its owner matching terminal acknowledgement and never becomes a rolled result',()=>setup({cancel:true},async f=>{
 await f.run();await f.ownerSettled;const writes=f.calls.updates.length;
 assert.deepEqual(await lateStage(f,{status:'cancelled',messageIdResult:undefined}),{status:'cancelled'});assert.deepEqual(await lateStage(f,{status:'uncertain',messageIdResult:undefined}),{status:'cancelled'});
 await assert.rejects(lateStage(f,{status:'done',messageIdResult:'invented-card'}));await assert.rejects(lateStage(f,{status:'cancelled',messageIdResult:'invented-card'}));assert.equal(f.calls.updates.length,writes);assert.equal(f.calls.dice,0);
}));
test('an uncertain terminal receipt only acknowledges uncertainty and cannot promote a failed native proof to done',()=>setup({wrongCard:card=>{card.rolls[0].options.degreeOfSuccess=0}},async f=>{
 await assert.rejects(f.run());await f.ownerSettled;const writes=f.calls.updates.length;
 assert.deepEqual(await lateStage(f,{status:'uncertain',messageIdResult:undefined}),{status:'uncertain'});assert.deepEqual(await lateStage(f,{status:'cancelled',messageIdResult:undefined}),{status:'uncertain'});
 await assert.rejects(lateStage(f,{status:'done',messageIdResult:'native-save'}));assert.equal(f.state().status,'uncertain');assert.equal(f.calls.updates.length,writes);assert.equal(f.calls.dice,1);
}));
for(const forged of ['user','nonce','source','card','public-binding','public-terminal','private-dc','handoff'])test(`a late ${forged} mismatch cannot use a cleared request or its authenticated completion`,()=>setup({},async f=>{
 await f.run();await f.ownerSettled;const changes={},state=f.state();let user=f.player;
 if(forged==='user')user=f.caster;
 if(forged==='nonce')changes.nonce='invented';
 if(forged==='source'){f.root.game.messages.set('copied-source',{...f.root.source,id:'copied-source',flags:structuredClone(f.root.source.flags)});changes.messageId='copied-source';}
 if(forged==='card'){f.root.game.messages.set('copied-card',{...f.root.game.messages.get('native-save'),id:'copied-card'});changes.messageIdResult='copied-card';}
 if(forged==='public-binding')state.binding.saveUserId=f.caster.id;
 if(forged==='public-terminal')state.status='uncertain';
 if(forged==='private-dc')f.root.game.messages.get('native-save').flags.pf2e.context.dc.value=99;
 if(forged==='handoff')f.users.activeGM=f.otherGM;
 const writes=f.calls.updates.length;await assert.rejects(lateStage(f,changes,user));assert.equal(f.calls.updates.length,writes);assert.equal(f.calls.dice,1);
}));
test('public source flags alone cannot invent a completed nonce even with a copied native proof',()=>setup({},async f=>{
 await f.run();await f.ownerSettled;const forged=structuredClone(f.state()),nonce='invented-completion';forged.binding.nonce=nonce;forged.messageId='forged-card';
 f.root.source.flags[ID].nativeTargetSaves[nonce]=forged;
 const old=f.root.game.messages.get('native-save'),card={...old,id:forged.messageId,flags:structuredClone(old.flags)};card.flags[ID].nativeTargetSave.nonce=nonce;card.flags.pf2e.context.options=card.flags.pf2e.context.options.map(option=>option.includes(':target-save:')?`${ID}:target-save:${nonce}`:option);f.root.game.messages.set(card.id,card);
 await assert.rejects(lateStage(f,{nonce,messageIdResult:card.id}));assert.equal(f.calls.dice,1);
}));
for(const mutation of ['disconnect','handoff','relink','delete-source'])test(`${mutation} closes the outstanding exact native window before a die and releases listeners`,()=>setup({hold:true},async f=>{
 const operation=f.run();await f.windowOpened;
 if(mutation==='disconnect'){f.player.active=false;for(const r of f.realms)r.Hooks.call('userConnected',f.player,false)}
 if(mutation==='handoff'){f.users.activeGM=f.otherGM;for(const r of f.realms)r.Hooks.call('updateUser',f.gm)}
 if(mutation==='relink')for(const r of f.realms){r.target.actor={...r.targetActor};r.game.actors.set(r.targetActor.id,r.target.actor);r.Hooks.call('updateToken',r.target)}
 if(mutation==='delete-source')for(const r of f.realms){r.game.messages.delete(r.source.id);r.Hooks.call('deleteChatMessage',r.source)}
 await assert.rejects(operation,/改变|确认/);await settled();assert.equal(f.calls.dice,0);assert.equal(f.root.game.messages.has('native-save'),false);assert.equal(f.remainingHooks(),0);
}));
test('the source item must be readable and real on the target player, without a fabricated GM save',()=>setup({},async f=>{f.owner.docs.delete(f.owner.item.uuid);await assert.rejects(f.run(),/改变/);await settled();assert.equal(f.calls.dice,0);assert.equal(f.remainingHooks(),0)}));
test('failure to reconstruct the original heightened overlay stops before using a base-rank save',()=>setup({},async f=>{f.owner.item.loadVariant=()=>null;await assert.rejects(f.run(),/改变|变体/);await settled();assert.equal(f.calls.dice,0);assert.equal(f.calls.native.length,0);assert.equal(f.remainingHooks(),0)}));
test('a saved spell save with another cast rank cannot settle the original paid spell',()=>setup({wrongCard:card=>{card.flags.pf2e.origin.castRank=1}},async f=>{await assert.rejects(f.run());await settled();assert.notEqual(f.state().status,'done');assert.equal(f.calls.dice,1);assert.equal(f.remainingHooks(),0)}));
test('a GM-published paid spell uses the original activity caster as the owner fallback without forging its author',()=>setup({},async f=>{
 f.realm(f.caster);f.player.character=null;
 for(const r of f.realms){r.source.author=f.gm;r.targetActor.testUserPermission=u=>['caster','target-player'].includes(u.id);r.game.messages.set('original-cast',{id:'original-cast',author:f.caster});}
 const result=await f.run();await settled();assert.equal(result.check.author.id,f.caster.id);assert.equal(f.root.source.author.id,f.gm.id);assert.equal(f.state().binding.sourceAuthorId,f.gm.id);assert.equal(f.calls.native[0].user,f.caster.id);assert.equal(f.remainingHooks(),0);
}));
for(const mode of ['public','self','blind'])test(`native ${mode} selection cannot widen a blind source audience`,()=>setup({blind:true,mode},async f=>{
 if(mode==='self'){await assert.rejects(f.run(),/受众|改变|确认/);assert.equal(f.root.game.messages.has('native-save'),false)}else{const result=await f.run();assert.equal(result.check.blind,true);assert.deepEqual(result.check.whisper,['gm']);assert.equal(result.check.flags.pf2e.context.messageMode,'blind')}
 await settled();assert.equal(f.remainingHooks(),0);
}));
for(const wrong of ['origin','dc','type','degree','target'])test(`a native card with altered ${wrong} cannot confirm the target-save operation`,()=>setup({wrongCard:card=>{
 if(wrong==='origin')card.flags.pf2e.origin.actor='Actor.other';if(wrong==='dc')card.flags.pf2e.context.dc.value=99;if(wrong==='type')card.flags.pf2e.context.type='skill-check';if(wrong==='degree')card.rolls[0].options.degreeOfSuccess=0;if(wrong==='target')card.flags.pf2e.context.target={actor:'Actor.other',token:'Scene.scene.Token.other'};
}},async f=>{await assert.rejects(f.run());await settled();assert.notEqual(f.state().status,'done');assert.equal(f.calls.dice,1);assert.equal(f.remainingHooks(),0)}));
test('a damage-taken Spiritual Scar source retains its real author and reaction item binding',()=>setup({scope:'spiritual-scar'},async f=>{const result=await f.run();await settled();assert.equal(result.check.flags.pf2e.origin.uuid,f.context.sourceItem.uuid);assert.equal(f.root.source.author.id,f.caster.id);assert.equal(f.calls.native[0].parameters.item.type,'action');assert.equal(f.calls.variants.length,0);assert.equal(f.remainingHooks(),0)}));
test('an untrusted player cannot ask another target owner to roll',()=>setup({},async f=>{const handler=f.owner.handlers.get('native-target-save:spell-combination');await assert.rejects(handler.call({socketdata:{userId:f.caster.id}},{messageId:'activity',nonce:'invented',request:f.request}),/改变/);assert.equal(f.calls.dice,0)}));

test('owner selection gives bound characters priority and never substitutes an online GM',async()=>{
 const gm={id:'gm',active:true,isGM:true},caster={id:'caster',active:true},bound={id:'bound',active:true,character:{uuid:'Actor.target'}},extra={id:'extra',active:true};const users=new Map([gm,caster,bound,extra].map(u=>[u.id,u]));
 const actor={id:'target',uuid:'Actor.target',testUserPermission:()=>true},game={user:gm,users};assert.equal(await selectNativeTargetSaveOwner({game,actor,casterUser:caster}),bound);
 bound.character=null;assert.equal(await selectNativeTargetSaveOwner({game,actor,casterUser:caster}),caster);caster.active=false;bound.active=false;assert.equal(await selectNativeTargetSaveOwner({game,actor}),extra);extra.active=false;await assert.rejects(selectNativeTargetSaveOwner({game,actor}),/不在线/);
});
test('multiple active owners require a meaningful GM choice rather than a first-owner guess',async()=>{
 const a={id:'a',active:true,name:'A'},b={id:'b',active:true,name:'B'},game={user:{id:'gm',isGM:true},users:new Map([[a.id,a],[b.id,b]])},actor={id:'pc',uuid:'Actor.pc',testUserPermission:()=>true};
 await assert.rejects(selectNativeTargetSaveOwner({game,actor}),/多名/);assert.equal(await selectNativeTargetSaveOwner({game,actor,choose:async({choices})=>{assert.deepEqual(choices.map(choice=>choice.value),['a','b']);return 'b'}}),b);
});

test('the actual PF2e saving-throw branch pins the roller token and foreign origin instead of the player selected target',()=>{
 const path=process.env.PF2E_NATIVE_BUNDLE;assert.ok(path,'the actual PF2e 8.5.1 source must be provided');const source=fs.readFileSync(path,'utf8').replace(/\s+/g,'');
 const start=source.indexOf('s=r===e.target||!e.target&&this.type===`saving-throw`'),end=source.indexOf(';',start);assert.ok(start>=0&&end>start);
 const actual=new Function('e','r','a','game',`let ${source.slice(start,end)};return {self:s,target:l,opposed:u};`);
 const roller={id:'target'},token={id:'bound-token'},selected={document:{id:'unrelated'}};const result=actual.call({type:'saving-throw',domains:['saving-throw']},{origin:{id:'caster'},item:{isOfType:()=>true},dc:{value:29}},roller,token,{user:{targets:{find:()=>selected}}});assert.deepEqual(result,{self:true,target:token,opposed:false});
 const rankExpression=source.match(/lete=c\?\.isOfType\(`spell`\)\?2\*c\.rank:c\?\.isOfType\(`physical`\)\?c\.level:f\?\.level\?\?m\.level/);assert.ok(rankExpression,'native incapacitation reads the reconstructed spell rank');
 const actualLevel=new Function('c','f','m',`${rankExpression[0].replace('lete=','let e=')};return e;`);assert.equal(actualLevel({rank:8,isOfType:type=>type==='spell'},null,{level:10}),16);
});
