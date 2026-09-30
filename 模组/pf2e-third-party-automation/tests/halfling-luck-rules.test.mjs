import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import vm from 'node:vm';
let api={};try{api=await import('../scripts/halfling-luck-rules.mjs')}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error}
const SOURCE='Compendium.pf2e.feats-srd.Item.ZbRVqf14RTJJIZXG';
const outcomes=['criticalFailure','failure','success','criticalSuccess'];
const json=value=>JSON.parse(JSON.stringify(value));

// The portable doubles use PF8.5.1's captured context and serialized dice shape.
// With both native paths, the entire eligibility matrix uses the actual installed
// CheckRoll and Foundry Die instead. Licensed source is never copied into tests.
async function loadNative(){
 const scope=vm.createContext({console,structuredClone,deepClone:structuredClone,_loc:v=>v});
 vm.runInContext(`globalThis.foundry={dice:{terms:{}},utils:{deepClone}};globalThis.CONFIG={Dice:{rolls:[],terms:{},termTypes:{},fulfillment:{methods:{},defaultMethod:'random'}}};Number.isNumeric=n=>typeof n==='number'&&Number.isFinite(n);Math.clamp=(n,a,b)=>Math.min(Math.max(n,a),b);`,scope);
 for(const[file,name]of [['roll.mjs','Roll'],['terms/term.mjs','RollTerm'],['terms/numeric.mjs','NumericTerm'],['terms/operator.mjs','OperatorTerm'],['terms/dice.mjs','DiceTerm'],['terms/die.mjs','Die'],['terms/function.mjs','FunctionTerm'],['terms/pool.mjs','PoolTerm']]){
  const text=(await readFile(path.join(process.env.FVTT_NATIVE_APP,'client/dice',file),'utf8')).replace(/^import .*?;\r?\n/gm,'').replace('export default class '+name,'globalThis.'+name+'=class '+name);
  vm.runInContext(text,scope,{filename:file});if(name!=='Roll')vm.runInContext(`foundry.dice.terms.${name}=${name};CONFIG.Dice.termTypes.${name}=${name};`,scope);
 }
 const source=await readFile(process.env.PF2E_NATIVE_BUNDLE,'utf8'),start=source.indexOf('xa = class CheckRoll extends Roll {'),end=source.indexOf(', StrikeAttackRoll =',start);
 assert.ok(start>=0&&end>start,'Review installed PF CheckRoll boundaries after an update');
 vm.runInContext('globalThis.CheckRoll='+source.slice(start+5,end)+';',scope);
 vm.runInContext('CONFIG.Dice.rolls=[Roll,CheckRoll];CONFIG.Dice.terms.d=Die;',scope);
 return {CheckRoll:scope.CheckRoll,Die:scope.Die,NumericTerm:scope.NumericTerm,OperatorTerm:scope.OperatorTerm};
}
const native=process.env.FVTT_NATIVE_APP&&process.env.PF2E_NATIVE_BUNDLE?await loadNative():null;
class Die{constructor(){this.number=1;this.faces=20;this.modifiers=[];this.results=[{result:4,active:true}];this._evaluated=true}}
class CheckRoll{
 constructor(){this.options={type:'skill-check',dice:'1d20',totalModifier:9,degreeOfSuccess:1,isReroll:false,rollerId:'owner'};this.dice=[new Die()];this.terms=this.dice;this._evaluated=true;this.total=13}
 get isReroll(){return this.options.isReroll??false}
 get isRerollable(){return !this.isReroll&&!this.dice.some(d=>d.modifiers.includes('kh')||d.modifiers.includes('kl'))}
 toJSON(){return {class:'CheckRoll',options:this.options,evaluated:this._evaluated,total:this.total,terms:this.dice.map(d=>({class:'Die',number:d.number,faces:d.faces,modifiers:d.modifiers,results:d.results,evaluated:d._evaluated}))}}
}
class CheckModifier{slug='athletics';totalModifier=9;modifiers=[]}
class Message{constructor(data){Object.assign(this,data)}get isCheckRoll(){return true}}
function roll(){
 if(!native)return new CheckRoll();
 const die=native.Die.fromData({class:'Die',number:1,faces:20,modifiers:[],results:[{result:4,active:true}],evaluated:true,options:{}});
 // Hydrate a deterministic recorded roll with the installed constructors and
 // terms. This tests native getters/serialization, not a browser dice throw.
 const result=new native.CheckRoll('',{},{type:'skill-check',dice:'1d20',totalModifier:9,degreeOfSuccess:1,isReroll:false,rollerId:'owner'});
 result.terms=[die,native.OperatorTerm.fromData({class:'OperatorTerm',operator:'+',evaluated:true}),native.NumericTerm.fromData({class:'NumericTerm',number:9,evaluated:true})];
 result._formula=result.resetFormula();result._evaluated=true;result._total=13;return result;
}
function fixture(t,{type='skill-check',degree=1}={}){
 const previous={CONFIG:globalThis.CONFIG,foundry:globalThis.foundry};
 globalThis.CONFIG={Dice:{rolls:[native?.CheckRoll??CheckRoll]},ChatMessage:{documentClass:Message}};globalThis.foundry={dice:{terms:{Die:native?.Die??Die}}};
 t.after(()=>{for(const[k,v]of Object.entries(previous)){if(v===undefined)delete globalThis[k];else globalThis[k]=v}});
 const user={id:'owner',active:true,isGM:false},actor={id:'a',uuid:'Actor.a',type:'character',isToken:false,canAct:true,isDead:false,items:new Map(),synthetics:{rollTwice:{}},testUserPermission:(u,p)=>u===user&&p==='OWNER'};
 const item={id:'luck',uuid:'Actor.a.Item.luck',actor,type:'feat',sourceId:SOURCE,system:{actionType:{value:'free'},frequency:{value:1,max:1,per:'day'},traits:{value:['fortune','halfling']},selfEffect:null},flags:{},_source:{system:{frequency:{max:1,per:'day'}}}};actor.items.set(item.id,item);
 const check=new CheckModifier(),r=roll();r.options.type=type;r.options.degreeOfSuccess=degree;
 const context={actor,token:null,origin:{actor,self:true,token:null},target:null,type,domains:['athletics','skill-check'],dc:{value:20},options:new Set(),rollTwice:false,substitutions:[],isReroll:false,messageMode:'public',traits:[],outcome:outcomes[degree],unadjustedOutcome:'failure',createMessage:false};
 if(type==='saving-throw'){context.origin=null;context.target={actor,self:true,token:null};context.domains=['reflex','saving-throw'];check.slug='reflex'}
 r.options.domains=[...context.domains];
 const game={system:{id:'pf2e',version:'8.5.1'},user,users:new Map([[user.id,user]]),actors:new Map([[actor.id,actor]]),pf2e:{CheckModifier,settings:{metagame:{results:true}}}};
 const card=new Message({author:user,speaker:{actor:actor.id},blind:false,whisper:[],rolls:[r],flags:{pf2e:{modifierName:check.slug,modifiers:[],context:{actor:actor.id,token:null,origin:context.origin?{actor:actor.uuid}:null,target:context.target?{actor:actor.uuid}:null,type,domains:[...context.domains],options:[],dc:{value:20},rollTwice:false,substitutions:[],isReroll:false,messageMode:'public',traits:[],outcome:outcomes[degree],unadjustedOutcome:'failure'}}}});
 return {game,actor,item,user,check,context,roll:r,card,requestedCreateMessage:true};
}
const assess=f=>{assert.equal(typeof api.assessHalflingLuck,'function');return api.assessHalflingLuck(f)};
const rejects=(f,reason)=>assert.deepEqual(assess(f),{eligible:false,reason});

test('exact source is recognized without matching a localized feat name',()=>{
 assert.equal(typeof api.isHalflingLuckItem,'function');assert.equal(api.isHalflingLuckItem({type:'feat',sourceId:SOURCE,name:'Renamed'}),true);
 assert.equal(api.isHalflingLuckItem({type:'feat',sourceId:'Compendium.other.Item.same',name:'Halfling Luck'}),false);assert.equal(api.isHalflingLuckItem({type:'action',sourceId:SOURCE}),false);
});
for(const[type,degree]of [['skill-check',0],['skill-check',1],['saving-throw',0],['saving-throw',1]])test(`${type} adjusted degree ${degree} can use its prepared daily free action`,t=>{
 const f=fixture(t,{type,degree});assert.deepEqual(assess(f),{eligible:true,reason:null});
 assert.equal(f.item._source.system.frequency.value,undefined);assert.equal(f.item.system.frequency.value,1);
});
test('zero reactions and another creatures turn do not restrict the free action',t=>{
 const f=fixture(t);f.game.combat={combatant:{actor:{uuid:'Actor.enemy'}}};f.actor.flags={'pf2e-reaction':{reaction:false}};
 const before=json(f.actor.flags);assert.equal(assess(f).eligible,true);assert.deepEqual(f.actor.flags,before);
});
test('native prepared contextual actor clones preserve the live owners own check',t=>{
 const f=fixture(t);const prepared={...f.actor,synthetics:{rollTwice:{}}};f.context.actor=prepared;f.context.origin.actor=prepared;assert.equal(assess(f).eligible,true);
});
for(const[label,mutate,reason]of [
 ['deleted actor',f=>f.game.actors.delete('a'),'actor-unavailable'],
 ['unlinked actor',f=>f.actor.isToken=true,'actor-unavailable'],
 ['NPC',f=>f.actor.type='npc','actor-unavailable'],
 ['missing owner',f=>f.actor.testUserPermission=()=>false,'not-owner'],
 ['disconnected user',f=>f.user.active=false,'not-owner'],
 ['cannot act',f=>f.actor.canAct=false,'cannot-act'],
 ['dead',f=>f.actor.isDead=true,'cannot-act'],
 ['deleted feat',f=>f.actor.items.clear(),'item-unavailable'],
 ['cloned feat',f=>f.item={...f.item},'item-unavailable'],
 ['other source',f=>f.item.sourceId='Compendium.other.Item.same','item-unavailable'],
 ['reaction cost',f=>f.item.system.actionType.value='reaction','unsupported-feat'],
 ['different recharge',f=>f.item.system.frequency.per='hour','unsupported-feat'],
 ['different maximum',f=>f.item.system.frequency.max=2,'unsupported-feat'],
 ['empty frequency',f=>f.item.system.frequency.value=0,'daily-use-unavailable'],
 ['unprepared frequency',f=>delete f.item.system.frequency.value,'daily-use-unavailable'],
 ['unknown system',f=>f.game.system.version='8.5.0','manual-native-compatibility'],
 ['other roller',f=>f.roll.options.rollerId='other','manual-native-evidence'],
 ['other actor check',f=>f.context.actor={uuid:'Actor.other'},'manual-native-evidence'],
 ['other card speaker',f=>f.card.speaker.actor='other','manual-native-evidence'],
 ['other card author',f=>f.card.author={id:'other'},'manual-native-evidence'],
 ['missing check',f=>f.check=null,'manual-native-evidence'],
 ['plain serialized roll',f=>f.roll=json(f.roll.toJSON()),'manual-native-evidence'],
 ['unresolved roll',f=>f.roll._evaluated=false,'manual-native-evidence'],
 ['missing DC',f=>f.context.dc=null,'manual-no-dc'],
 ['inconsistent DC',f=>f.card.flags.pf2e.context.dc.value=21,'manual-native-evidence'],
 ['inconsistent outcome',f=>f.card.flags.pf2e.context.outcome='success','manual-native-evidence'],
 ['missing adjusted degree',f=>delete f.roll.options.degreeOfSuccess,'manual-native-evidence'],
])test(label+' does not authorize luck',t=>{const f=fixture(t);mutate(f);rejects(f,reason)});
for(const type of ['attack-roll','initiative','flat-check','check'])test(type+' is not a qualifying trigger',t=>rejects(fixture(t,{type}),'not-skill-or-save'));
for(const degree of [2,3])test('an adjusted '+outcomes[degree]+' cannot use an unadjusted failure',t=>rejects(fixture(t,{degree}),'not-failed'));

for(const[label,mutate,reason]of [
 ['context reroll',f=>f.context.isReroll=true,'already-rerolled'],
 ['roll reroll',f=>f.roll.options.isReroll=true,'already-rerolled'],
 ['draft reroll',f=>f.card.flags.pf2e.context.isReroll=true,'already-rerolled'],
 ['reroll option',f=>f.context.options.add('check:reroll'),'already-rerolled'],
 ['context fortune',f=>f.context.options.add('fortune'),'fortune-occupied'],
 ['draft fortune',f=>f.card.flags.pf2e.context.options.push('fortune'),'fortune-occupied'],
 ['selected substitution',f=>f.context.substitutions=[{slug:'assurance',selected:true,effectType:'fortune',value:10}],'manual-substitution'],
 ['draft substitution',f=>f.card.flags.pf2e.context.substitutions=[{slug:'custom',required:true,selected:true,effectType:'misfortune',value:1}],'manual-substitution'],
 ['context misfortune',f=>f.context.options.add('misfortune'),'manual-fortune-misfortune'],
 ['draft misfortune',f=>f.card.flags.pf2e.context.options.push('misfortune'),'manual-fortune-misfortune'],
 ['keep lower',f=>f.context.rollTwice='keep-lower','manual-fortune-misfortune'],
 ['keep higher',f=>f.context.rollTwice='keep-higher','fortune-occupied'],
 ['both traits',f=>{f.context.options.add('fortune');f.context.options.add('misfortune')},'manual-fortune-misfortune'],
 ['kh die',f=>f.roll.dice[0].modifiers.push('kh'),'manual-nonordinary-d20'],
 ['kl die',f=>f.roll.dice[0].modifiers.push('kl'),'manual-nonordinary-d20'],
 ['extra die',f=>f.roll.dice[0].number=2,'manual-nonordinary-d20'],
 ['not d20',f=>f.roll.dice[0].faces=12,'manual-nonordinary-d20'],
 ['modified die',f=>f.roll.dice[0].modifiers.push('r1'),'manual-nonordinary-d20'],
 ['discarded die',f=>f.roll.dice[0].results[0].discarded=true,'manual-nonordinary-d20'],
])test(label+' cannot silently become an ordinary luck reroll',t=>{const f=fixture(t);mutate(f);rejects(f,reason)});
test('opposing matching synthetic sources cannot hide behind native rollTwice=false',t=>{
 const f=fixture(t);f.context.actor={...f.actor,synthetics:{rollTwice:{athletics:[{keep:'higher',predicate:{test:()=>true}},{keep:'lower',predicate:{test:()=>true}}]}}};f.context.origin.actor=f.context.actor;
 rejects(f,'manual-fortune-misfortune');
});
test('nonmatching synthetic predicates and unselected substitutions do not occupy the roll',t=>{
 const f=fixture(t);f.actor.synthetics.rollTwice.athletics=[{keep:'higher',predicate:{test:()=>false}},{keep:'lower',predicate:{test:()=>false}}];
 f.context.substitutions=[{selected:false,effectType:'fortune'}];assert.equal(assess(f).eligible,true);
});
for(const[label,mutate,reason]of [
 ['secret option',f=>f.context.options.add('secret'),'manual-private-check'],
 ['secret trait',f=>f.card.flags.pf2e.context.traits=['secret'],'manual-private-check'],
 ['blind mode',f=>f.context.messageMode='blind','manual-private-check'],
 ['GM-only draft',f=>f.card.flags.pf2e.context.messageMode='gm','manual-private-check'],
 ['whispered draft',f=>f.card.whisper=['gm'],'manual-private-check'],
 ['blind draft',f=>f.card.blind=true,'manual-private-check'],
 ['hidden results',f=>f.game.pf2e.settings.metagame.results=false,'manual-private-check'],
 ['unknown message mode',f=>delete f.card.flags.pf2e.context.messageMode,'manual-unknown-privacy'],
 ['unknown draft visibility',f=>delete f.card.blind,'manual-unknown-privacy'],
])test(label+' cannot disclose a failed check through an automatic prompt',t=>{const f=fixture(t);mutate(f);rejects(f,reason)});
test('a hidden DC alone is distinct from hidden result visibility',t=>{const f=fixture(t);f.context.dc.visible=false;f.card.flags.pf2e.context.dc.visible=false;assert.equal(assess(f).eligible,true)});
test('installed native CheckRoll and Die qualify with their evaluated getters',{skip:!native},t=>{
 const f=fixture(t);assert.equal(f.roll.constructor.name,'CheckRoll');assert.equal(f.roll.total,13);assert.equal(f.roll.isRerollable,true);assert.equal(assess(f).eligible,true);
});
test('a separately hydrated native card roll must match the captured roll',t=>{
 const f=fixture(t);f.card.rolls=[roll()];f.card.rolls[0].options.domains=[...f.context.domains];assert.equal(assess(f).eligible,true);
 f.card.rolls[0].options.degreeOfSuccess=0;rejects(f,'manual-native-evidence');
});
test('a save belongs to its target actor even when an enemy is the origin',t=>{
 const f=fixture(t,{type:'saving-throw'});f.context.origin={actor:{uuid:'Actor.enemy'},self:false};f.card.flags.pf2e.context.origin={actor:'Actor.enemy'};
 assert.equal(assess(f).eligible,true);f.card.flags.pf2e.context.target.actor='Actor.other';rejects(f,'manual-native-evidence');
});
function attachToken(f){
 const scene={id:'scene',tokens:new Map()},token={documentName:'Token',id:'token',uuid:'Scene.scene.Token.token',actorLink:true,actor:f.actor,parent:scene};scene.tokens.set(token.id,token);f.game.scenes=new Map([[scene.id,scene]]);
 f.context.token=token;f.context.origin.token=token;f.card.speaker={actor:'a',scene:'scene',token:'token'};f.card.flags.pf2e.context.token='token';f.card.flags.pf2e.context.origin.token=token.uuid;return token;
}
test('the linked token must still be its exact scene document and own native speaker',t=>{
 const f=fixture(t),token=attachToken(f);assert.equal(assess(f).eligible,true);
 token.parent.tokens.set('token',{...token});rejects(f,'manual-native-evidence');
});
test('an unrelated token cannot supply the same actor check proof',t=>{
 const f=fixture(t);attachToken(f);f.card.flags.pf2e.context.origin.token='Scene.scene.Token.other';rejects(f,'manual-native-evidence');
});
test('an unknown synthetic predicate fails closed without throwing or mutating the result',t=>{
 const f=fixture(t);f.actor.synthetics.rollTwice.athletics=[{keep:'higher',predicate:{test(){throw Error('unknown predicate')}}}];const before=json(f.roll.toJSON());
 rejects(f,'manual-native-evidence');assert.deepEqual(json(f.roll.toJSON()),before);
});
test('missing original domains cannot prove that matching synthetic sources were considered',t=>{
 const f=fixture(t);delete f.context.domains;rejects(f,'manual-native-evidence');
});
test('different domains on the captured card cannot authorize this check',t=>{
 const f=fixture(t);f.card.flags.pf2e.context.domains=['deception','skill-check'];rejects(f,'manual-native-evidence');
});
test('a public native draft cannot prove that a no-message callers own destination is public',t=>{
 const f=fixture(t);f.requestedCreateMessage=false;rejects(f,'manual-unproven-draft-privacy');
});
test('without an original caller snapshot createMessage false stays manual',t=>{
 const f=fixture(t);delete f.requestedCreateMessage;rejects(f,'manual-unproven-draft-privacy');
});
test('an ordinary publication request can use an internal native draft',t=>{
 const f=fixture(t);assert.equal(f.context.createMessage,false);assert.equal(assess(f).eligible,true);
 delete f.requestedCreateMessage;f.context.createMessage=true;assert.equal(assess(f).eligible,true);
});
test('ordinary privacyProof fields cannot bypass unproven no-message privacy',t=>{
 const f=fixture(t);f.requestedCreateMessage=false;
 for(const proof of [true,'public',{private:false}]){f.context.privacyProof=proof;rejects(f,'manual-unproven-draft-privacy')}
});
