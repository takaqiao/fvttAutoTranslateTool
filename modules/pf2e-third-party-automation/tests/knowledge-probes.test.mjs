import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
let probes;
try {probes=await import('../scripts/knowledge-probes.mjs');} catch {}
class Message {constructor(data){Object.assign(this,data);this.id=null;}getFlag(ns,key){return key.split('.').reduce((o,k)=>o?.[k],this.flags[ns]);}}
const globals={ChatMessage:Message};
function fixture(){
 const actor={id:'a',uuid:'Actor.a',rules:[]},check={slug:'society',totalModifier:0,modifiers:[{slug:'proficiency',label:'Proficiency',modifier:9,enabled:true,toObject(){return {slug:this.slug,modifier:this.modifier};}}],calculateTotal(options){assert.ok(options.has('action:recall-knowledge'));this.totalModifier=options.has('target:trait:humanoid')?13:9;}};
 const context={actor,origin:{actor},type:'skill-check',domains:['skill-check','society'],options:new Set(['action:recall-knowledge','action:recall-knowledge:society','target:trait:humanoid']),title:'Society',createMessage:false};
 return {actor,check,context};
}
test('locally bound knowledge probe captures native calculated modifiers without rolling or afterRoll',async()=>{
 assert.ok(probes?.withKnowledgeProbe,'local native probe capability is missing');const f=fixture();let native=0,callback=0;
 const result=await probes.withKnowledgeProbe({actor:f.actor,statistic:'society',globals},async marker=>{
  f.context.options.add(marker);return probes.interceptKnowledgeProbe(()=>{native++;throw Error('probe rolled dice');},f.check,f.context,null,(roll,outcome,message)=>{callback++;assert.equal(roll.options.totalModifier,13);assert.equal(message.id,null);assert.equal(message.getFlag('pf2e','context.type'),'skill-check');assert.ok(!message.getFlag('pf2e','context.options').some(x=>x.includes('knowledge-probe:')));assert.match(message.flavor,/Proficiency/);});
 });
 assert.equal(result.result,null);assert.equal(result.receipt.captured,true);assert.equal(result.receipt.check,f.check);assert.equal(native,0);assert.equal(callback,1);
});
test('copied probe roll option outside its live local scope cannot suppress a real native check',async()=>{
 assert.ok(probes?.withKnowledgeProbe);const f=fixture();let stale;
 await probes.withKnowledgeProbe({actor:f.actor,statistic:'society',globals},async marker=>{stale=marker;});f.context.options.add(stale);let native=0;
 const result=await probes.interceptKnowledgeProbe(()=>{native++;return 'native';},f.check,f.context);assert.equal(result,'native');assert.equal(native,1);
});
test('a live capability cannot be reused for another actor or a non-RK check',async()=>{
 assert.ok(probes?.withKnowledgeProbe);const f=fixture();let native=0;
 await probes.withKnowledgeProbe({actor:f.actor,statistic:'society',globals},async marker=>{f.context.options.add(marker);f.context.actor={uuid:'Actor.other'};await probes.interceptKnowledgeProbe(()=>{native++;},f.check,f.context);f.context.actor=f.actor;f.context.options.delete('action:recall-knowledge');await probes.interceptKnowledgeProbe(()=>{native++;},f.check,f.context);});assert.equal(native,2);
});
const nativeBundle=process.env.FVTT_PF2E_BUNDLE??'';
test('installed native StatisticCheck boundary skips afterRoll when the captured Check returns null',{skip:!fs.existsSync(nativeBundle)},async()=>{
 assert.ok(probes?.withKnowledgeProbe);const bundle=fs.readFileSync(nativeBundle,'utf8'),start=bundle.indexOf('D = await Sa.roll(E, w, null, e.callback);'),end=bundle.indexOf('\n\t}\n\tget breakdown()',start);assert.ok(start>0&&end>start);
 const body=bundle.slice(start+'D = '.length,end).replace('await Sa.roll(E, w, null, e.callback);','const D = await Sa.roll(E, w, null, e.callback);');
 const run=new Function('Sa','E','w','e','m','i','y',`return (async()=>{${body}})();`),f=fixture();let native=0,after=0;f.actor.rules=[{afterRoll(){after++;}}];
 await probes.withKnowledgeProbe({actor:f.actor,statistic:'society',globals},async marker=>{f.context.options.add(marker);assert.equal(await run({roll:(...args)=>probes.interceptKnowledgeProbe(()=>{native++;return {total:99};},...args)},f.check,f.context,{callback(){}},f.actor,f.context.domains,f.context.options),null);});assert.equal(native,0);assert.equal(after,0);
});
function nativeAfterRoll(bundle,className){
 const start=bundle.indexOf(`${className} = class`),method=bundle.indexOf('async afterRoll(',start),brace=bundle.indexOf(') {',method)+2;assert.ok(start>0&&method>start&&brace>method);
 let depth=1,end=brace+1;for(;depth&&end<bundle.length;end++){if(bundle[end]==='{')depth++;else if(bundle[end]==='}')depth--;}
 return bundle.slice(method,end);
}
test('actual native FlatModifier removeAfterRoll uses the captured modifier rule identity once after claim',{skip:!fs.existsSync(nativeBundle)},async()=>{
 const bundle=fs.readFileSync(nativeBundle,'utf8'),afterRoll=new Function(`return ({${nativeAfterRoll(bundle,'FlatModifierRuleElement')}}).afterRoll;`)();let deletes=0;
 const flags={},message={flags,async update(changes){for(const[key,value]of Object.entries(changes)){let object=this;const path=key.split('.');for(const part of path.slice(0,-1))object=object[part]??={};object[path.at(-1)]=value;}}};
 const rule={afterRoll,removeAfterRoll:'if-enabled',item:{isOfType:type=>type==='effect',async delete(){assert.equal(flags['pf2e-third-party-automation'].workbenchRecall.probeUse.status,'claimed');deletes++;}}},actor={rules:[rule]},check={modifiers:[{rule,enabled:true}]},receipt={captured:true,actor,check,domains:['skill-check','society'],primaryContext:{options:new Set(['action:recall-knowledge'])}};
 const candidate={statistic:'society',targetUuid:'Scene.s.Token.enemy'},roll={total:24,dice:[{modifiers:[]}]};await probes.consumeKnowledgePrimary({message,candidate,receipt,roll});await probes.consumeKnowledgePrimary({message,candidate,receipt,roll});assert.equal(deletes,1);
});
test('actual native RollTwice and SubstituteRoll afterRoll receive their used dice and substitutions',{skip:!fs.existsSync(nativeBundle)},async()=>{
 const bundle=fs.readFileSync(nativeBundle,'utf8'),game={pf2e:{settings:{automation:{removeEffects:true}}}},twice=new Function('game',`return ({${nativeAfterRoll(bundle,'RollTwiceRuleElement')}}).afterRoll;`)(game),substitute=new Function(`return ({${nativeAfterRoll(bundle,'SubstituteRollRuleElement')}}).afterRoll;`)();let deletes=0;
 const actor={items:new Map([['effect',{}]])},item={id:'effect',rules:[],isOfType:()=>true,async delete(){deletes++;}},rule={actor,item,selector:['skill-check'],test:()=>true,removeAfterRoll:'if-enabled',slug:'native-substitution'};
 await twice.call(rule,{domains:['skill-check'],roll:{dice:[{modifiers:['kh']}]},rollOptions:new Set()});assert.equal(deletes,1);
 await substitute.call(rule,{roll:{dice:[]},context:{substitutions:[{slug:rule.slug,selected:true}]}});assert.equal(deletes,2);
 await substitute.call(rule,{roll:{dice:[{modifiers:[]}]},context:{substitutions:[{slug:rule.slug,selected:false}]}});assert.equal(deletes,2);
});
