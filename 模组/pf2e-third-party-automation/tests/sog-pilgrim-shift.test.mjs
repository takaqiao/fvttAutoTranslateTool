import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createPilgrimShift} from '../scripts/sog-pilgrim-shift.mjs';

const M='pf2e-third-party-automation',PACK='pf2e.equipment-srd';
const clone=value=>structuredClone(value);
function merge(target,changes) {
 for(const [path,value]of Object.entries(changes)){
  const parts=path.split('.');let at=target;
  for(const key of parts.slice(0,-1))at=at[key]??={};
  const key=parts.at(-1);
  if(key.startsWith('-='))delete at[key.slice(2)];
  else if(value&&typeof value==='object'&&!Array.isArray(value))merge(at[key]??={},value);
  else at[key]=clone(value);
 }
}
function weapon(id='longsword',changes={}) {
 const data={_id:id,name:'长剑',type:'weapon',system:{slug:id,baseItem:id,category:'martial',group:'sword',level:{value:0},damage:{dice:1,die:'d8',damageType:'slashing',modifier:0},traits:{value:['versatile-p'],rarity:'common',otherTags:[]},usage:{value:'held-in-one-hand'},range:null,reload:{value:null},runes:{potency:0,striking:0,property:[]},material:{type:null,grade:null},specific:null}};
 merge(data,changes);
 return {id,uuid:`Compendium.${PACK}.Item.${id}`,type:data.type,name:data.name,system:clone(data.system),get isMelee(){return !this.system.range},get hands(){return {'held-in-one-hand':'1','held-in-one-plus-hands':'1+','held-in-two-hands':'2'}[this.system.usage.value]??'1'},get category(){return this.system.category},toObject:()=>clone(data)};
}
function fixture(forms=[weapon()]) {
 const actor={id:'a',uuid:'Actor.a',type:'character',items:new Map()};
 const data={_id:'branch',name:'大杉枝',type:'weapon',img:'branch.webp',flags:{world:{sogWontonNativeAutomation:{key:'branch-of-the-great-sugi'},sogB2Ch2Resources:{originalSlug:'branch-of-the-great-sugi',fileSha256:'a'.repeat(64)}},[M]:{sogPilgrim:{spent:true}},pf2e:{custom:true}},_stats:{compendiumSource:'Item.original'},system:{slug:'branch-of-the-great-sugi',description:{value:'大杉枝的说明'},publication:{title:'Pilgrim'},level:{value:6},price:{value:{gp:250}},baseItem:'staff',category:'simple',group:'club',damage:{dice:1,die:'d4',damageType:'bludgeoning',modifier:0,persistent:{number:1,faces:4,type:'fire'}},traits:{value:['magical','two-hand-d8'],rarity:'rare',otherTags:['pilgrim'],toggles:{versatile:{selected:'fire'}},config:{modular:[{damageType:'fire',traits:[]}]}},usage:{value:'held-in-one-hand',canBeAmmo:true},range:30,maxRange:120,reload:{value:'1'},meleeUsage:{group:'axe',damage:{die:'d6',type:'slashing'},traits:['sweep']},ammo:{baseType:'rounds',capacity:1},expend:1,selectedAmmoId:'old-ammo',attribute:'dex',bonus:{value:2},splashDamage:{value:2},equipped:{carryType:'held',handsHeld:1,invested:null},runes:{potency:1,striking:1,property:['ghostTouch']},specific:{price:{gp:250},runes:{potency:1,striking:1,property:['ghostTouch']}},material:{type:null,grade:null}}};
 const writes=[],item={id:'branch',uuid:'Actor.a.Item.branch',actor,...clone(data),_source:clone(data),toObject(){return clone(this._source)},async update(patch){writes.push(clone(patch));merge(this._source,patch);this.system=clone(this._source.system);this.flags=clone(this._source.flags);if(this.system.damage.dice===1)this.system.damage.dice+=this.system.runes.striking;return this}};
 actor.items.set(item.id,item);
 let reads=0;
 const pack={async getIndex(){reads++;return forms.map(form=>form.toObject())}};
 const game={actors:new Map([['a',actor]]),scenes:new Map(),packs:new Map([[PACK,pack]])};
 const documents=new Map(forms.map(form=>[form.uuid,form]));
 const options={game,fromUuid:async uuid=>documents.get(uuid)};
 return {item,actor,game,forms,writes,options,shift:createPilgrimShift(options),reads:()=>reads};
}
function dialog(t,callback) {
 const previous=globalThis.foundry;
 globalThis.foundry={applications:{api:{DialogV2:{wait:callback}}}};
 t.after(()=>{globalThis.foundry=previous});
}

test('shift rebuilds a longsword and a dagger on the same reward without old weapon mechanics',async()=>{
 const dagger=weapon('dagger',{name:'匕首','system.category':'simple','system.damage.die':'d4','system.damage.damageType':'piercing','system.traits.value':['agile','finesse','thrown-10','versatile-s'],'system.reload.value':'-'});
 const f=fixture([weapon(),dagger]),identity=clone(f.item._source);
 assert.equal(await f.shift.apply({item:f.item,formUuid:f.forms[0].uuid}),f.item);
 assert.equal(f.actor.items.get('branch'),f.item);
 assert.equal(f.item.system.baseItem,'longsword');
 assert.equal(f.item.system.category,'martial');
 assert.equal(f.item.system.group,'sword');
 assert.deepEqual(f.item._source.system.damage,{dice:1,die:'d8',damageType:'slashing',modifier:0});
 assert.deepEqual(f.item.system.traits.value,['versatile-p','magical']);
 for(const key of ['range','maxRange','meleeUsage','ammo','expend','selectedAmmoId','attribute'])assert.equal(f.item.system[key],null,key);
 assert.equal(f.item.system.reload.value,null);
 assert.equal(f.item.system.usage.canBeAmmo,false);
 assert.deepEqual(f.item.system.bonus,{value:0});
 assert.deepEqual(f.item.system.splashDamage,{value:0});
 assert.deepEqual(f.item._source.system.traits.toggles,{});
 assert.deepEqual(f.item._source.system.traits.config,{});
 await f.shift.apply({item:f.item,formUuid:dagger.uuid});
 assert.deepEqual(f.item._source.system.damage,{dice:1,die:'d4',damageType:'piercing',modifier:0});
 assert.deepEqual(f.item.system.traits.value,['agile','finesse','thrown-10','versatile-s','magical']);
 assert.equal(f.item.system.reload.value,'-');
 for(const key of ['_id','name','type','img','flags','_stats'])assert.deepEqual(f.item._source[key],identity[key],key);
 for(const key of ['slug','description','publication','level','price','equipped','runes','specific','material'])assert.deepEqual(f.item._source.system[key],identity.system[key],key);
 assert.equal(f.item.system.traits.rarity,'rare');
 assert.deepEqual(f.item.system.traits.otherTags,['pilgrim']);
 assert.equal(f.writes.length,2);
});

test('shift copies raw one-die damage rather than prepared striking dice',async()=>{
 const form=weapon();form.system.damage.dice=2;
 const f=fixture([form]);
 await f.shift.apply({item:f.item,formUuid:form.uuid});
 assert.equal(f.item._source.system.damage.dice,1);
 assert.equal(f.item.system.damage.dice,2);
 await f.shift.apply({item:f.item,formUuid:form.uuid});
 assert.equal(f.item._source.system.damage.dice,1);
 assert.equal(f.item.system.damage.dice,2);
 assert.deepEqual(f.item.system.runes,{potency:1,striking:1,property:['ghostTouch']});
});

test('a missing baseItem is filled only from a registered native base weapon slug',async t=>{
 const previous=globalThis.CONFIG;globalThis.CONFIG={PF2E:{baseWeaponTypes:{longsword:'PF2E.Weapon.Base.longsword'}}};
 t.after(()=>{globalThis.CONFIG=previous});
 const f=fixture([weapon('longsword',{'system.baseItem':null})]);
 dialog(t,async config=>{
  assert.ok(config.content.includes(f.forms[0].uuid));
  return f.forms[0].uuid;
 });
 assert.equal(await f.shift.select(f.item),f.forms[0].uuid);
 await f.shift.apply({item:f.item,formUuid:f.forms[0].uuid});
 assert.equal(f.item.system.baseItem,'longsword');
 const unknown=fixture([weapon('unknown',{'system.baseItem':null})]);
 await assert.rejects(unknown.shift.apply({item:unknown.item,formUuid:unknown.forms[0].uuid}));
 assert.equal(unknown.writes.length,0);
});

test('shift rejects unsuitable documents and foreign UUIDs before updating',async t=>{
 const cases=[
  ['two hands',{'system.usage.value':'held-in-two-hands'}],
  ['one plus hands',{'system.usage.value':'held-in-one-plus-hands'}],
  ['ranged',{'system.range':30}],
  ['unarmed',{'system.category':'unarmed'}],
  ['not weapon',{type:'equipment'}],
  ['magic',{'system.traits.value':['magical']}],
  ['runes',{'system.runes.property':['ghostTouch']}],
  ['valuable material',{'system.material.type':'silver','system.material.grade':'low'}],
  ['higher level',{'system.level.value':1}],
  ['specific weapon',{'system.specific':{price:{gp:5}}}],
  ['not base weapon',{'system.slug':'custom-longsword'}],
 ];
 for(const [name,changes]of cases)await t.test(name,async()=>{
  const f=fixture([weapon('longsword',changes)]);
  await assert.rejects(f.shift.apply({item:f.item,formUuid:f.forms[0].uuid}));
  assert.equal(f.writes.length,0);
 });
 for(const uuid of ['Item.longsword','Compendium.third-party.weapons.Item.longsword','Compendium.pf2e.equipment-srd.Item.longsword.extra'])await t.test(uuid,async()=>{
  const f=fixture();let resolved=false;
  const shift=createPilgrimShift({...f.options,fromUuid:async()=>{resolved=true;return f.forms[0]}});
  await assert.rejects(shift.apply({item:f.item,formUuid:uuid}));
  assert.equal(resolved,false);
  assert.equal(f.writes.length,0);
 });
});

test('shift uses native prepared melee and hands validation, and rejects an incorrect resolution',async()=>{
 for(const change of [form=>Object.defineProperty(form,'isMelee',{value:false}),form=>Object.defineProperty(form,'hands',{value:'2'}),form=>{form.uuid='Item.other'}]){
  const f=fixture();change(f.forms[0]);
  await assert.rejects(f.shift.apply({item:f.item,formUuid:`Compendium.${PACK}.Item.longsword`}));
  assert.equal(f.writes.length,0);
 }
});

test('shift requires the live held branch and rejects tree form even if still held',async()=>{
 for(const change of [
  f=>{f.item.flags.world.sogWontonNativeAutomation.key='spirit-fan'},
  f=>{f.item.system.equipped.carryType='worn'},
  f=>{f.item.system.equipped.handsHeld=0},
  f=>{f.actor.items.set(f.item.id,{...f.item})},
  f=>{f.game.actors.set(f.actor.id,{...f.actor})},
  f=>{f.actor.items.set('tree',{type:'effect',flags:{[M]:{sogPilgrim:{kind:'tree',source:f.item.uuid}}}})},
 ]){
  const f=fixture();change(f);
  await assert.rejects(f.shift.apply({item:f.item,formUuid:f.forms[0].uuid}));
  assert.equal(f.writes.length,0);
 }
});

test('shift checks possession again after the target loads',async()=>{
 const f=fixture(),shift=createPilgrimShift({...f.options,fromUuid:async()=>{f.actor.items.delete(f.item.id);return f.forms[0]}});
 await assert.rejects(shift.apply({item:f.item,formUuid:f.forms[0].uuid}));
 assert.equal(f.writes.length,0);
});

test('shift rejects an update that did not save the new weapon form',async()=>{
 const f=fixture();f.item.update=async()=>undefined;
 await assert.rejects(f.shift.apply({item:f.item,formUuid:f.forms[0].uuid}));
 assert.equal(f.item.system.baseItem,'staff');
});

test('shift accepts a live synthetic token actor and refuses it after removal',async()=>{
 const f=fixture(),scene={id:'s',tokens:new Map()},token={id:'t',parent:scene,actor:f.actor};
 f.actor.isToken=true;f.actor.token=token;scene.tokens.set(token.id,token);f.game.scenes.set(scene.id,scene);f.game.actors.clear();
 await f.shift.apply({item:f.item,formUuid:f.forms[0].uuid});
 scene.tokens.delete(token.id);
 await assert.rejects(f.shift.apply({item:f.item,formUuid:f.forms[0].uuid}));
 assert.equal(f.writes.length,1);
});

test('select lists all legal native weapon names with escaped markup and no hundred-entry limit',async t=>{
 const forms=Array.from({length:125},(_,i)=>weapon(`weapon${i}`,{name:i===124?'<匕首 & "刃">':`武器${i}`}));
 forms.push(weapon('bad',{'system.usage.value':'held-in-two-hands'}));
 const f=fixture(forms);let shown=0;
 dialog(t,async config=>{
  shown++;
  assert.equal((config.content.match(/<option\b/g)??[]).length,125);
  assert.ok(config.content.includes('&lt;匕首 &amp; &quot;刃&quot;&gt;'));
  assert.ok(!config.content.includes('Item.bad'));
  assert.equal(config.rejectClose,false);
  const confirm=config.buttons.find(button=>button.action==='shift');
  return confirm.callback(null,{form:{elements:{formUuid:{value:forms[124].uuid}}}});
 });
 assert.equal(await f.shift.select(f.item),forms[124].uuid);
 assert.equal(await f.shift.select(f.item),forms[124].uuid);
 assert.equal(f.reads(),1);
 assert.equal(shown,2);
 assert.equal(f.writes.length,0);
});

test('cancel and closing the native selection do not update the reward',async t=>{
 const f=fixture();
 dialog(t,async config=>config.buttons.find(button=>button.action==='cancel').callback());
 assert.equal(await f.shift.select(f.item),null);
 globalThis.foundry.applications.api.DialogV2.wait=async()=>null;
 assert.equal(await f.shift.select(f.item),null);
 assert.equal(f.writes.length,0);
});

test('select rejects forged choices and a branch released while its dialog was open',async t=>{
 const f=fixture();
 dialog(t,async()=>`Compendium.${PACK}.Item.unlisted`);
 await assert.rejects(f.shift.select(f.item));
 globalThis.foundry.applications.api.DialogV2.wait=async()=>{f.item.system.equipped.handsHeld=0;return f.forms[0].uuid};
 await assert.rejects(f.shift.select(f.item));
 assert.equal(f.writes.length,0);
});
