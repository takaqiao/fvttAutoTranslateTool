import test from 'node:test';
import assert from 'node:assert/strict';
import {createPilgrimAreas} from '../scripts/sog-pilgrim-areas.mjs';
import {MODULE_ID} from '../scripts/rules.mjs';
import {PILGRIM_FLAG} from '../scripts/sog-pilgrim-rules.mjs';

function update(target, changes) {
 for(const [path,value] of Object.entries(changes)){
  const parts=path.split('.');let object=target;
  for(const part of parts.slice(0,-1))object=object[part]??={};
  object[parts.at(-1)]=structuredClone(value);
 }
}
function fixture(key='hairpin') {
 let id=0;
 const game={system:{id:'pf2e'},world:{id:'sog'},user:{id:'gm',isGM:true},users:{activeGM:{id:'gm'}},time:{worldTime:100},scenes:new Map(),messages:new Map(),combat:null,modules:new Map([['sequencer',{active:true}]])};
 const scene={id:'scene',uuid:'Scene.scene',grid:{size:100,distance:5,isSquare:true,measurePath:points=>({distance:Math.hypot(points[1].x-points[0].x,points[1].y-points[0].y)/20})},templates:new Map(),regions:new Map(),tiles:new Map(),tokens:new Map()};
 game.scenes.set(scene.id,scene);
 const documents=new Map();
 const collections={MeasuredTemplate:'templates',Region:'regions',Tile:'tiles'};
 scene.createEmbeddedDocuments=async(type,rows)=>rows.map(row=>{
  const document={...structuredClone(row),id:`doc${++id}`,parent:scene};
  document.uuid=`${scene.uuid}.${type}.${document.id}`;
  document.update=async changes=>update(document,changes);
  scene[collections[type]].set(document.id,document);documents.set(document.uuid,document);
  return document;
 });
 scene.deleteEmbeddedDocuments=async(type,ids)=>{for(const id of ids){const doc=scene[collections[type]].get(id);scene[collections[type]].delete(id);documents.delete(doc?.uuid);}};
 const actor={id:'actor',uuid:'Actor.actor',items:new Map(),getReach:()=>5,testUserPermission:()=>true};
 actor.createEmbeddedDocuments=async(type,rows)=>rows.map(row=>{
  assert.equal(type,'Item');
  const effect={...structuredClone(row),id:`effect${++id}`,actor,isExpired:false};
  effect.uuid=`${actor.uuid}.Item.${effect.id}`;
  effect.update=async changes=>update(effect,changes);
  actor.items.set(effect.id,effect);documents.set(effect.uuid,effect);
  return effect;
 });
 actor.deleteEmbeddedDocuments=async(type,ids)=>{assert.equal(type,'Item');for(const id of ids){const doc=actor.items.get(id);actor.items.delete(id);documents.delete(doc?.uuid);}};
 const slug=key==='hairpin'?'hairpin-of-blooming-flowers':'branch-of-the-great-sugi';
 const item={id:'reward',uuid:'Actor.actor.Item.reward',name:key==='hairpin'?'繁花发簪 Hairpin of Blooming Flowers':'大杉枝 Branch of the Great Sugi',type:key==='hairpin'?'equipment':'weapon',actor,system:{equipped:{carryType:key==='hairpin'?'worn':'held',handsHeld:1,invested:true}},flags:{world:{sogWontonNativeAutomation:{key:slug},sogB2Ch2Resources:{originalSlug:slug,fileSha256:'a'.repeat(64)}}},update:async changes=>update(item,changes)};
 actor.items.set(item.id,item);documents.set(item.uuid,item);
 const source={id:'source',uuid:'Scene.scene.Token.source',parent:scene,actor,x:50,y:50,width:1,height:1,elevation:0,object:{center:{x:100,y:100}}};
 const target={id:'target',uuid:'Scene.scene.Token.target',parent:scene,actor:{uuid:'Actor.target',isOfType:(...types)=>types.includes('npc'),traits:new Set(['undead']),system:{attributes:{hp:{value:0}}}},x:150,y:50,width:1,height:1,elevation:0,object:{center:{x:200,y:100}},hidden:true};
 scene.tokens.set(source.id,source);scene.tokens.set(target.id,target);documents.set(source.uuid,source);documents.set(target.uuid,target);
 const effects={tree:{name:'效果：杉树形态 Effect: Sugi Tree Form',type:'effect',system:{duration:{value:1,unit:'hours',expiry:'turn-start'},start:{value:0,initiative:null}},flags:{}},storm:{name:'效果：花瓣风暴 Effect: Petal Storm',type:'effect',system:{duration:{value:1,unit:'minutes',expiry:'turn-start'},start:{value:0,initiative:null}},flags:{}}};
 const fromUuid=async uuid=>uuid==='Item.1ade2944acadc195'?{toObject:()=>structuredClone(effects.tree),type:'effect'}:uuid==='Item.9570410bfd571f7d'?{toObject:()=>structuredClone(effects.storm),type:'effect'}:documents.get(uuid);
 const canvas={scene:{id:'gm-view',tokens:new Map()},grid:{measurePath:()=>({distance:Infinity})}};
 const errors=[];const media={area:async()=>{},clearArea:async()=>{}};
 const card=async options=>{const message={id:`message${++id}`,...options,flags:{[MODULE_ID]:{[PILGRIM_FLAG]:{generated:true,source:options.item.uuid,nonce:options.nonce}}}};game.messages.set(message.id,message);return message;};
 const announce=async({item,nonce,text})=>{const message={id:`message${++id}`,text,flags:{[MODULE_ID]:{[PILGRIM_FLAG]:{announce:true,source:item.uuid,nonce}}}};game.messages.set(message.id,message);return message;};
 const callbacks=new Map();const Hooks={on:(name,fn)=>callbacks.set(name,fn)};
 const areas=createPilgrimAreas({game,fromUuid,canvas,media,onError:error=>errors.push(error)});
 const emit=async event=>callbacks.get('tpaSogPilgrimAreaEvent')?.(event);
 return {game,scene,item,actor,source,target,canvas,fromUuid,media,card,announce,areas,errors,Hooks,emit};
}
const own=document=>document.flags?.[MODULE_ID]?.[PILGRIM_FLAG];

test('a cancelled storm placement leaves equipment and scene untouched',async()=>{
 const f=fixture();f.canvas.scene=f.scene;globalThis.Sequencer={Crosshair:{show:async()=>undefined}};
 assert.equal(await f.areas.pick({item:f.item,source:f.source}),null);
 assert.equal(f.scene.templates.size,0);assert.equal(f.actor.items.size,1);
 assert.equal(f.item.system.equipped.carryType,'worn');
});

test('storm range uses the token scene grid while the GM views another scene',()=>{
 const f=fixture();
 assert.doesNotThrow(()=>f.areas.validatePlacement({item:f.item,source:f.source,position:{x:1300,y:100},range:60}));
 assert.throws(()=>f.areas.validatePlacement({item:f.item,source:f.source,position:{x:1320,y:100},range:60}),/范围/);
 f.scene.tokens.delete(f.source.id);
 assert.throws(()=>f.areas.validatePlacement({item:f.item,source:f.source,position:{x:100,y:100},range:60}),/场景/);
});

test('placement needs the requester to view the token scene, not the GM',async()=>{
 const f=fixture();globalThis.Sequencer={Crosshair:{show:async()=>({x:400,y:100})}};
 await assert.rejects(f.areas.pick({item:f.item,source:f.source}),/场景/);
 f.canvas.scene=f.scene;
 assert.deepEqual(await f.areas.pick({item:f.item,source:f.source}),{x:400,y:100});
});

test('storm creates a timed effect and native movement region without creating a creature',async()=>{
 const f=fixture();
 const result=await f.areas.storm({item:f.item,source:f.source,position:{x:200,y:100},nonce:'storm123',card:f.card,announce:f.announce});
 assert.ok(result?.effectId);assert.equal(f.scene.templates.size,1);assert.equal(f.scene.tokens.size,2);
 const template=[...f.scene.templates.values()][0];assert.equal(template.distance,15);assert.equal(template.t,'circle');
 const effect=f.actor.items.get(result.effectId);assert.equal(effect.system.start.value,100);assert.equal(effect.system.duration.unit,'minutes');assert.equal(effect.system.duration.expiry,'turn-start');
 const region=[...f.scene.regions.values()][0];assert.deepEqual(region.behaviors[0].system.events,['tokenMoveIn','tokenTurnStart']);
 assert.deepEqual(region.shapes,[{type:'circle',x:200,y:100,radius:300,gridBased:true}]);
 assert.equal(own(effect).source,f.item.uuid);assert.equal(own(region).nonce,'storm123');
 assert.equal(f.game.messages.size,1);const message=[...f.game.messages.values()][0];
 assert.equal(own(message).announce,true);assert.equal(own(message).nonce,'storm123');assert.equal(message.formula,undefined);
 assert.match(message.text,/花瓣风暴/);assert.match(message.text,/1 分钟/);assert.match(message.text,/DC 23/);
});

test('a storm whose announcement fails leaves no area behind',async()=>{
 const f=fixture();
 await assert.rejects(f.areas.storm({item:f.item,source:f.source,position:{x:200,y:100},nonce:'storm123',card:f.card,announce:async()=>{throw Error('chat offline');}}),/chat offline/);
 assert.equal(f.actor.items.size,1);assert.equal(f.scene.regions.size,0);assert.equal(f.scene.templates.size,0);assert.equal(f.game.messages.size,0);
});

test('tree becomes a static tree and drops the weapon while preserving its held mode',async()=>{
 const f=fixture('branch');
 const result=await f.areas.tree({item:f.item,source:f.source,target:f.target,nonce:'tree1234',card:f.card,announce:f.announce});
 assert.ok(result?.effectId);assert.equal(f.scene.tokens.size,2);assert.equal(f.scene.tiles.size,1);
 assert.equal(f.item.system.equipped.carryType,'dropped');assert.equal(f.item.system.equipped.handsHeld,0);
 const effect=f.actor.items.get(result.effectId);assert.equal(own(effect).equippedBefore.carryType,'held');assert.equal(effect.system.duration.unit,'hours');
 const [message,notice]=[...f.game.messages.values()];assert.equal(message.formula,'(3d8+8)[healing]');assert.equal(message.gm,false);assert.equal(message.target,undefined);
 assert.match(message.label,/能看见/);assert.match(message.label,/更高/);
 assert.equal(own(notice).announce,true);assert.equal(own(notice).nonce,'tree1234');assert.match(notice.text,/余烬重生/);assert.match(notice.text,/1 小时/);
 await f.areas.restore({item:f.item,user:f.game.user});
 assert.equal(f.item.system.equipped.carryType,'held');assert.equal(f.item.system.equipped.handsHeld,1);
 assert.equal(f.actor.items.size,1);assert.equal(f.scene.templates.size,0);assert.equal(f.scene.tiles.size,0);
});

test('ending a tree does not overwrite a later equipment change or unrelated scene resource',async()=>{
 const f=fixture('branch');await f.areas.tree({item:f.item,source:f.source,target:f.target,nonce:'tree1234',card:f.card,announce:f.announce});
 f.scene.tiles.set('unrelated',{id:'unrelated',flags:{}});f.item.system.equipped.carryType='stowed';
 await f.areas.restore({item:f.item,user:f.game.user});
 assert.equal(f.item.system.equipped.carryType,'stowed');assert.equal(f.scene.tiles.size,1);assert.ok(f.scene.tiles.has('unrelated'));
});

test('the top-down tree stays centered on the corpse with a 15-foot texture on different grids',async()=>{
 for(const distance of [5,10]){
  const f=fixture('branch');f.scene.grid.distance=distance;
  await f.areas.tree({item:f.item,source:f.source,target:f.target,nonce:`tree-grid-${distance}`,card:f.card,announce:f.announce});
  const tile=[...f.scene.tiles.values()][0],size=f.scene.grid.size*15/distance;
  assert.deepEqual({x:tile.x,y:tile.y},f.target.object.center);
  assert.deepEqual({width:tile.width,height:tile.height},{width:size,height:size});
  assert.deepEqual({x:tile.texture.anchorX,y:tile.texture.anchorY},{x:0.5,y:0.5});
  assert.equal(tile.texture.src,'modules/pf2e-season-of-ghosts/assets/maps/other/treetops/tree-top01narchy.webp');
  assert.equal([...f.scene.templates.values()][0].distance,15);
 }
});

test('failed native card creation rolls tree resources and equipment back',async()=>{
 const f=fixture('branch');
 await assert.rejects(f.areas.tree({item:f.item,source:f.source,target:f.target,nonce:'tree1234',card:async()=>{throw Error('card failure');},announce:f.announce}),/card failure/);
 assert.equal(f.actor.items.size,1);assert.equal(f.scene.tiles.size,0);assert.equal(f.scene.templates.size,0);assert.equal(f.item.system.equipped.carryType,'held');
});

test('area media failure preserves the completed native rules',async()=>{
 const f=fixture();f.media.area=async()=>{throw Error('media failure');};
 const result=await f.areas.storm({item:f.item,source:f.source,position:{x:200,y:100},nonce:'storm123',card:f.card,announce:f.announce});
 assert.ok(f.actor.items.has(result.effectId));assert.equal(f.game.messages.size,1);assert.equal(f.errors[0]?.message,'media failure');
});

test('each real storm entry gets one card and duplicate movement survives reconnect',async()=>{
 const f=fixture();await f.areas.storm({item:f.item,source:f.source,position:{x:200,y:100},nonce:'storm123',card:f.card,announce:f.announce});f.areas.register({Hooks:f.Hooks,card:f.card});
 const region=[...f.scene.regions.values()][0];const event={name:'tokenMoveIn',region,data:{token:f.target,movement:{id:'move1',origin:{x:0,y:100},destination:{x:200,y:100},passed:{waypoints:[{x:200,y:100}]}}}};
 await f.emit(event);await f.emit(event);assert.equal(f.game.messages.size,2);
 const reconnected=createPilgrimAreas({game:f.game,fromUuid:f.fromUuid,canvas:f.canvas,media:f.media});reconnected.register({Hooks:f.Hooks,card:f.card});
 await f.emit(event);assert.equal(f.game.messages.size,2);
 await f.emit({...event,data:{...event.data,movement:{...event.data.movement,id:'move2'}}});assert.equal(f.game.messages.size,3);
 const messages=[...f.game.messages.values()].slice(1);for(const message of messages){assert.equal(message.formula,'{1d10[slashing],1d10[vitality]}');assert.deepEqual(message.save,{type:'reflex',dc:23});assert.equal(message.gm,true);assert.equal(message.target,f.target);}
 await f.emit({...event,name:'tokenEnter'});assert.equal(f.game.messages.size,3);
});

test('storm damage on a player character is public so its owner can roll the save',async()=>{
 const f=fixture();await f.areas.storm({item:f.item,source:f.source,position:{x:200,y:100},nonce:'storm123',card:f.card,announce:f.announce});f.areas.register({Hooks:f.Hooks,card:f.card});
 const region=[...f.scene.regions.values()][0],event={name:'tokenTurnStart',region,data:{combat:{uuid:'Combat.combat'},round:1,turn:0}};
 f.target.actor.hasPlayerOwner=true;await f.emit({...event,data:{...event.data,token:f.target}});
 f.target.actor.hasPlayerOwner=false;await f.emit({...event,data:{...event.data,turn:1,token:f.target}});
 assert.deepEqual([...f.game.messages.values()].slice(1).map(message=>message.gm),[false,true]);
});

test('storm turn start is separate from entry, deduplicated, and ignores skipped turns',async()=>{
 const f=fixture();await f.areas.storm({item:f.item,source:f.source,position:{x:200,y:100},nonce:'storm123',card:f.card,announce:f.announce});f.areas.register({Hooks:f.Hooks,card:f.card});
 const region=[...f.scene.regions.values()][0];const event={name:'tokenTurnStart',region,data:{token:f.target,combat:{uuid:'Combat.combat'},round:1,turn:2,skipped:false}};
 await f.emit(event);await f.emit(event);assert.equal(f.game.messages.size,2);
 await f.emit({...event,data:{...event.data,round:2,skipped:true}});assert.equal(f.game.messages.size,2);
 await f.emit({...event,data:{...event.data,round:2}});assert.equal(f.game.messages.size,3);
});

test('native expiry removes only the resources linked to that effect',async()=>{
 const f=fixture();const result=await f.areas.storm({item:f.item,source:f.source,position:{x:200,y:100},nonce:'storm123',card:f.card,announce:f.announce});
 f.game.time.worldTime=1000;await f.areas.reconcile(f.actor);assert.ok(f.actor.items.has(result.effectId));
 f.actor.items.get(result.effectId).isExpired=true;await f.areas.reconcile(f.actor);
 assert.equal(f.actor.items.size,1);assert.equal(f.scene.regions.size,0);assert.equal(f.scene.templates.size,0);
});

test('non-primary GM and stale token events cannot create a storm card',async()=>{
 const f=fixture();await f.areas.storm({item:f.item,source:f.source,position:{x:200,y:100},nonce:'storm123',card:f.card,announce:f.announce});f.areas.register({Hooks:f.Hooks,card:f.card});
 const event={name:'tokenTurnStart',region:[...f.scene.regions.values()][0],data:{token:f.target,combat:{uuid:'Combat.combat'},round:1,turn:2}};
 f.game.user.id='other';await f.emit(event);assert.equal(f.game.messages.size,1);
 f.game.user.id='gm';f.scene.tokens.delete(f.target.id);await f.emit(event);assert.equal(f.game.messages.size,1);
});

test('manually deleting a tree effect restores the weapon and removes its display',async()=>{
 const f=fixture('branch');const result=await f.areas.tree({item:f.item,source:f.source,target:f.target,nonce:'tree1234',card:f.card,announce:f.announce});
 const effect=f.actor.items.get(result.effectId);await f.actor.deleteEmbeddedDocuments('Item',[effect.id]);
 await f.areas.deleted(effect);
 assert.equal(f.item.system.equipped.carryType,'held');assert.equal(f.scene.tiles.size,0);assert.equal(f.scene.templates.size,0);
});

test('removing a reward removes its active region and effect',async()=>{
 const f=fixture();await f.areas.storm({item:f.item,source:f.source,position:{x:200,y:100},nonce:'storm123',card:f.card,announce:f.announce});
 await f.actor.deleteEmbeddedDocuments('Item',[f.item.id]);await f.areas.deleted(f.item);
 assert.equal(f.actor.items.size,0);assert.equal(f.scene.regions.size,0);assert.equal(f.scene.templates.size,0);
});

test('a stale effect source and a forged region event do not apply damage',async()=>{
 const f=fixture();const result=await f.areas.storm({item:f.item,source:f.source,position:{x:200,y:100},nonce:'storm123',card:f.card,announce:f.announce});f.areas.register({Hooks:f.Hooks,card:f.card});
 const region=[...f.scene.regions.values()][0];own(region).nonce='othernonce';
 await f.emit({name:'tokenTurnStart',region,data:{token:f.target,combat:{uuid:'Combat.combat'},round:1,turn:2}});assert.equal(f.game.messages.size,1);
 await f.actor.deleteEmbeddedDocuments('Item',[f.item.id]);await f.areas.reconcile(f.actor);
 assert.equal(f.actor.items.has(result.effectId),false);
});
