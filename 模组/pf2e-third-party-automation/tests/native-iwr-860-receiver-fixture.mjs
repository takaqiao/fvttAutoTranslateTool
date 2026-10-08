import vm from 'node:vm';
import {flatRegion,observerRegion} from '../scripts/native-source-shapes.mjs';
import {native860} from './native-iwr-860-fixture.mjs';

export function receiverContext(source){
 const r=native860.receiverRegions,game={pf2e:{settings:{variants:{abp:'noABP'}}}},context=vm.createContext({game,
  N:value=>!!value&&typeof value==='object'&&!Array.isArray(value),v:value=>!!value&&typeof value==='object'&&!Array.isArray(value),
  foundry:{utils:{deepClone:structuredClone,getProperty(){throw Error('injection-forbidden')}}},
  _loc:value=>value,sluggify:value=>String(value),CONFIG:{PF2E:{damageTypes:{},abilities:{}}},setHasElement:(set,value)=>set.has(value),objectHasKey:(value,key)=>Object.hasOwn(value,key),M:values=>[...new Set(values)],ErrorPF2e:message=>Error(message)});
 vm.runInContext('Math.clamp=(v,min,max)=>Math.min(Math.max(v,min),max);const '+r.predicate+';const AutomaticBonusProgression$1=class{'+r.abpEnabled+r.abpSuppression+'};',context);
 // Foundry DataModel supplies these prepared fields; resolution methods and
 // all Predicate, Modifier and FlatModifier operations are actual 8.6 source.
 vm.runInContext('const q=class{constructor(raw,{parent,sourceIndex}){Object.assign(this,{ignored:false,invalid:false,type:"untyped",force:false,battleForm:false,fromEquipment:true,critical:null,removeAfterRoll:false,tags:[],hideIfDisabled:false},raw);this.predicate=new Un(raw.predicate??[]);this.parent=parent;this.sourceIndex=sourceIndex}get item(){return this.parent}get actor(){return this.parent.actor}getReducedLabel(){return this.item.name}failValidation(){this.ignored=true}'+r.ruleResolution+'#replaceFormulaData(){throw Error("dynamic-value")} };'+r.modifier,context);
 vm.runInContext('const FlatModifierRuleElement=class extends q{'+r.flatConstructor+'get selectors(){return this.selector}'+flatRegion(source)+'};',context);
 vm.runInContext('game.pf2e.variantRules={AutomaticBonusProgression:AutomaticBonusProgression$1};'+observerRegion(source,'pf2e').region,context);
 return context;
}
