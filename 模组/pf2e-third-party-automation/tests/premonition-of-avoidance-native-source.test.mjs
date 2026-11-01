import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {Predicate} from './knowledge-pf2e-861-fixture.mjs';
import {createPremonitionOfAvoidance} from '../scripts/premonition-of-avoidance.mjs';

const input=process.env.BOB2_CURRENT_PF2E_SOURCE??process.env.PF2E_NATIVE_861_BUNDLE;
const source=input?readFileSync(input,'utf8'):null;
const OPTION='premonition-of-avoidance',SOURCE='Compendium.pf2e.feats-srd.Item.KYl1rWFOHe0e6VqJ';
function region(start,end){
 const from=source.indexOf(start),to=source.indexOf(end,from);
 assert.ok(from>=0&&to>from,'native source region: '+start);
 return source.slice(from,to);
}
function nativeFixture({choice='use',higherBonus=0,hazard=true}={}){
 const foundry={utils:{deepClone:structuredClone,mergeObject:Object.assign}},CONFIG={PF2E:{damageTypes:{}},ChatMessage:{modes:{public:0}}};
 const objectHasKey=(object,key)=>Object.hasOwn(object,key),game={world:{id:'bob-2'},system:{id:'pf2e'},user:{settings:{showCheckDialogs:false}},actors:new Map(),pf2e:{settings:{metagame:{secretChecks:false}}},settings:{get:()=> 'public'}};
 const classes=region('var jc =','function createAttributeModifier')+region('var HIGHER_BONUS =','}, Mc = class DamageDicePF2e')+'};';
 const {Modifier,StatisticModifier,CheckModifier}=new Function('foundry','Wn','_loc','sluggify','setHasElement','objectHasKey','CONFIG','j','o','v','ui',classes+'return {Modifier:Y,StatisticModifier,CheckModifier};')(
  foundry,Predicate,value=>value,value=>value,(set,value)=>set.has(value),objectHasKey,CONFIG,values=>[...new Set(values)],value=>value!=null,value=>!!value&&typeof value==='object',{notifications:{error:message=>assert.fail(message)}});
 // Execute the actual Check.roll entry through predicate recalculation and its
 // modifiers dialog. Dice, chat creation and rendering belong to runtime QA.
 const prefix=region('static async roll(e, t = {}, n = null, r) {','\n\t\tlet s = [], c =');
 class CheckModifiersDialog{constructor(_check,resolve){this.resolve=resolve}render(){this.resolve(false)}}
 const roll=new Function('game','foundry','CONFIG','objectHasKey','CheckModifiersDialog',`return ({${prefix.replace('static async roll','async roll')}return e.totalModifier;}}).roll;`)(game,foundry,CONFIG,objectHasKey,CheckModifiersDialog);
 const item={id:'feat',uuid:'Actor.hero.Item.feat',type:'feat',sourceId:SOURCE,system:{rules:[{key:'RollOption',domain:'saving-throw',option:OPTION,toggleable:true,value:true}]}};
 const actor={id:'hero',uuid:'Actor.hero',type:'character',items:new Map([['feat',item]]),testUserPermission:()=>true,toggleRollOption:async(domain,option,id,value)=>{assert.equal(domain,'saving-throw');assert.equal(option,OPTION);assert.equal(id,'feat');item.system.rules[0].value=value}};game.actors.set(actor.id,actor);
 const modifier=new Modifier({slug:OPTION,label:'Premonition of Avoidance',modifier:2,type:'circumstance',predicate:['origin:type:hazard',OPTION]});
 const modifiers=[new Modifier({slug:'base',label:'Reflex',modifier:10,type:'untyped'}),modifier];
 if(higherBonus)modifiers.push(new Modifier({slug:'higher',label:'Other circumstance',modifier:higherBonus,type:'circumstance'}));
 const options=new Set(['saving-throw',OPTION,hazard?'origin:type:hazard':'origin:type:npc']);
 const statistic=new StatisticModifier('reflex',modifiers,options),check=new CheckModifier('reflex',statistic,[],options);
 const context={actor,type:'saving-throw',options,domains:['all','saving-throw','reflex'],skipDialog:true};
 const provider=createPremonitionOfAvoidance({game,choose:async()=>choice});
 return {provider,item,actor,context,check,roll};
}

for(const [choice,want]of [['use',12],['decline',10],[null,10]])test('native prepared hazard modifier is recalculated after '+String(choice),{skip:!source},async()=>{
 const f=nativeFixture({choice});assert.equal(f.check.totalModifier,12);
 assert.equal(await f.provider.interceptCheck(f.roll,f.check,f.context),want);
 assert.equal(f.check.modifiers.find(modifier=>modifier.slug===OPTION).enabled,choice==='use');
 assert.equal(f.item.system.rules[0].value,false);
});

test('native circumstance stacking keeps a higher +3 bonus instead of adding +2',{skip:!source},async()=>{
 const f=nativeFixture({higherBonus:3});
 assert.equal(await f.provider.interceptCheck(f.roll,f.check,f.context),13);
 assert.equal(f.check.modifiers.find(modifier=>modifier.slug===OPTION).enabled,false);
 assert.equal(f.item.system.rules[0].value,false);
});

test('native predicate does not grant the armed bonus to a non-hazard save',{skip:!source},async()=>{
 const f=nativeFixture({hazard:false});
 assert.equal(await f.provider.interceptCheck(f.roll,f.check,f.context),10);
 assert.equal(f.item.system.rules[0].value,true);
});

test('a second native check with already prepared options loses the consumed bonus',{skip:!source},async()=>{
 const f=nativeFixture(),secondContext={...f.context,options:new Set(f.context.options)};
 assert.equal(await f.provider.interceptCheck(f.roll,f.check,f.context),12);
 assert.equal(await f.provider.interceptCheck(f.roll,f.check,secondContext),10);
});

test('cancelling the native modifiers dialog returns null and disarms the actor',{skip:!source},async()=>{
 const f=nativeFixture();f.context.skipDialog=false;
 assert.equal(await f.provider.interceptCheck(f.roll,f.check,f.context),null);
 assert.equal(f.item.system.rules[0].value,false);
});
