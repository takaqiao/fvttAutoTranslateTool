import test from 'node:test';
import assert from 'node:assert/strict';
import {existsSync,readFileSync} from 'node:fs';
import vm from 'node:vm';
let api={};try{api=await import('../scripts/metapower/card.mjs')}catch(e){if(e.code!=='ERR_MODULE_NOT_FOUND')throw e}
const snapshot=(changes={})=>({kind:'siphoning',powerId:'electric-surge',level:5,discharge:true,siphon:{applies:true},area:{type:'line',baseDistance:60,distance:30},...changes});
test('selected discharge formula and area are immutable across renders and ordinary scaling is retained',()=>{
 assert.equal(typeof api.cardLinkPlan,'function');const s=snapshot(),links=[{kind:'damage',baseFormula:'6d4[electricity]'},{kind:'area',type:'line',distance:20},{kind:'effect',uuid:'Compendium.battlezoo-eldamon-pf2e.conditions.Bi2aHykg6CZrQCnR'},{kind:'area',type:'line',distance:40},{kind:'damage',baseFormula:'6d4[electricity]'}];
 const a=api.cardLinkPlan(s,links);assert.equal(a[0].disabled,true);assert.equal(a[4].formula,'(1+5)d8[electricity]');assert.equal(a[2].disabled,true);assert.equal(a[3].distance,30);
 assert.deepEqual(api.cardLinkPlan(s,links),a);
});
test('fixed-failure damage stays fixed and chain uses confirmed triggering damage',()=>{
 const staticPlan=api.cardLinkPlan(snapshot({powerId:'static-shock',discharge:false}),[{kind:'damage'},{kind:'damage'},{kind:'damage'}]);assert.equal(staticPlan[2].formula,'(2+5)[electricity]');assert.equal(staticPlan[1].disabled,true);
 const chain=api.cardLinkPlan(snapshot({powerId:'reactive-chain',triggerDamage:19}),[{kind:'damage'}]);assert.equal(chain[0].formula,'9[electricity]');
});
test('Widen never suppresses native effect links or expands a nonmatching shape',()=>{
 const links=[{kind:'area',type:'line',distance:30},{kind:'area',type:'emanation',distance:10},{kind:'effect',uuid:'x'}];
 const plan=api.cardLinkPlan(snapshot({kind:'widen',siphon:{applies:false},discharge:false,area:{type:'line',distance:40}}),links);
 assert.equal(plan[0].distance,40);assert.equal(plan[1].distance,10);assert.equal(plan[2].disabled,undefined);
});
test('High Voltage has no immediate damage link on its activation card',()=>{
 const plan=api.cardLinkPlan(snapshot({powerId:'high-voltage'}),[{kind:'damage',baseFormula:'4d6[electricity]'}]);assert.equal(plan[0].disabled,true);
});
test('Electric Shot keeps fixed failure and offers the selected branch base for Shocked half-failure',()=>{
 const plan=api.cardLinkPlan(snapshot({powerId:'electric-shot'}),[{kind:'damage'},{kind:'damage'},{kind:'damage'}]);
 assert.equal(plan[2].formula,'5[electricity]');assert.equal(plan[2].shockedFailureFormula,'(2+5)d8[electricity]');
});

function renderedSave(t){
 const previous=globalThis.document;t.after(()=>{if(previous===undefined)delete globalThis.document;else globalThis.document=previous});
 globalThis.document={createElement:()=>({setAttribute(){}})};
 const check={dataset:{pf2Check:'fortitude',against:'eldamon',rollerRole:'target',pf2Traits:'electricity',pf2RollOptions:'damaging-effect,existing-option'}};
 const root={dataset:{},append(){},querySelector:()=>null,querySelectorAll:selector=>selector==='[data-pf2-check]'?[check]:[]};
 const message={id:'native-card'},receipt={nonce:'channel',snapshot:snapshot({kind:'normal',powerId:'anvil-crawler-lightning',discharge:false,siphon:{applies:false}})};
 api.renderMetapowerCard(message,root,{receipt});api.renderMetapowerCard(message,root,{receipt});
 return {check,marker:'pf2e-third-party-automation:metapower:native-card:channel'};
}

test('rendered saves preserve native PF2e check options and add one original-channel marker',t=>{
 const {check,marker}=renderedSave(t);
 assert.deepEqual(check.dataset.pf2RollOptions.split(','),['damaging-effect','existing-option',marker]);
 assert.equal(check.dataset.rollOptions,undefined,'check links must not use the damage-link option attribute');
});

const nativePath=process.env.PF2E_NATIVE_BUNDLE??'';
test('actual PF2e inline-save handler carries the rendered channel marker into the native check',{skip:!existsSync(nativePath)},async t=>{
 const {check,marker}=renderedSave(t),source=readFileSync(nativePath,'utf8');
 const start=source.indexOf('static async #onClickInlineCheck(e, t) {'),end=source.indexOf('\n\tstatic #onClickInlineTemplate',start);
 assert.ok(start>=0&&end>start,'Native inline-check handler shape changed');
 const code=source.slice(start,end).replace('static async #onClickInlineCheck','async function');
 let rolled;
 const origin={uuid:'Actor.origin',getStatistic:slug=>slug==='eldamon'?{dc:{label:'Power DC'},label:'Eldamon'}:null,isOfType:()=>false};
 const item={actor:origin,slug:'anvil-crawler-lightning',isOfType:(...types)=>types.includes('feat')};
 const target={uuid:'Actor.target',getStatistic:()=>({roll:params=>{rolled=params}})};
 const context={resolveActorAndItemFromHTML:()=>({actor:origin,item,sheetActor:null}),getSelectedActors:()=>[target],tupleHasValue:(a,v)=>a.includes(v),Br:['fortitude','reflex','will'],splitListString:s=>s.split(',').filter(Boolean),M:a=>[...new Set(a)],o:Boolean,eventToRollParams:()=>({skipDialog:true}),sluggify:s=>s,CONFIG:{PF2E:{actionTraits:{electricity:'Electricity'}}},game:{user:{targets:{first:()=>null}}},console};
 const handler=vm.runInNewContext('('+code+')',context);await handler({},check);
 assert.ok(rolled,'The native saving-throw statistic was not invoked');
 assert.ok(rolled.extraRollOptions.includes(marker),'Actual native check lost the original channel marker');
 assert.ok(rolled.extraRollOptions.includes('damaging-effect'));assert.ok(rolled.extraRollOptions.includes('existing-option'));
 assert.equal(rolled.dc.slug,'eldamon');assert.equal(rolled.origin,origin);assert.equal(rolled.item,item);
});

function renderedChain(t,{existing=false,source='Compendium.battlezoo-eldamon-pf2e.powers.Item.fzV5Ly3a9nEsfcAJ'}={}){
 const previous=globalThis.document;t.after(()=>{if(previous===undefined)delete globalThis.document;else globalThis.document=previous});
 const element=()=>({dataset:{},className:'',setAttribute(){},classList:{contains:()=>false}});
 globalThis.document={createElement:element};
 const anchors=existing?[Object.assign(element(),{dataset:{damageRoll:'',formula:'1[electricity]',baseFormula:'1[electricity]'}})]:[];
 const root={dataset:{},append(a){if('damageRoll'in a.dataset)anchors.push(a)},querySelector:selector=>selector==='a.inline-roll[data-damage-roll]'?anchors[0]:['.card-content','.message-content'].includes(selector)?root:null,querySelectorAll:selector=>selector==='[data-pf2-check]'?[]:anchors};
 const receipt={nonce:'chain-use',sourceUuid:source,itemUuid:'Actor.caster.Item.chain',snapshot:snapshot({kind:'normal',powerId:'reactive-chain',powerSourceUuid:source,triggerDamage:8,discharge:false,siphon:{applies:false},traits:['concentrate','electricity','magical']})};
 api.renderMetapowerCard({id:'chain-card'},root,{receipt});api.renderMetapowerCard({id:'chain-card'},root,{receipt});
 return {anchors,receipt};
}

test('original Chain card without published damage gains one immutable native link from its confirmed amount',t=>{
 const {anchors}=renderedChain(t);assert.equal(anchors.length,1);
 const [a]=anchors;assert.equal(a.dataset.formula,'4[electricity]');assert.equal(a.dataset.baseFormula,'4[electricity]');assert.ok('damageRoll'in a.dataset);assert.ok('immutable'in a.dataset);
 assert.equal(a.dataset.itemUuid,'Actor.caster.Item.chain');assert.equal(a.dataset.rollOptions,'pf2e-third-party-automation:metapower:chain-card:chain-use');
 assert.equal(a.dataset.traits,'concentrate,electricity,magical');
});
test('Chain rendering never duplicates an existing damage link or invents one for a different source',t=>{
 assert.equal(renderedChain(t,{existing:true}).anchors.length,1);
 assert.equal(renderedChain(t,{source:'Compendium.other.powers.Item.unreviewed'}).anchors.length,0);
});
test('actual PF2e inline-damage handler sends the generated Chain link through native damage context',{skip:!existsSync(nativePath)},async t=>{
 const {anchors}=renderedChain(t);assert.equal(anchors.length,1);const [anchor]=anchors,source=readFileSync(nativePath,'utf8');
 const start=source.indexOf('static async _onClickInlineRoll(e) {'),end=source.indexOf('\n\tstatic processUserVisibility',start);assert.ok(start>=0&&end>start,'Native damage-link handler shape changed');
 class Item{}const origin={uuid:'Actor.caster'},item=Object.assign(new Item(),{uuid:'Actor.caster.Item.chain',name:'Reactive Chain',isOfType:()=>true});
 const message={speakerActor:origin,getRollData:()=>({actor:origin,item})};let augmented,nativeRoll;
 const context={htmlClosest:(_el,selector)=>selector==='a'?anchor:selector==='li.chat-message'?{dataset:{messageId:'chain-card'}}:null,ui:{windows:{}},game:{messages:new Map([['chain-card',message]]),settings:{get:()=> 'public'}},Z:Item,CONFIG:{ChatMessage:{modes:{public:'public'}}},ChatMessagePF2e:{getSpeaker:()=>({actor:'caster'})},objectHasKey:(o,k)=>Object.hasOwn(o,k),splitListString:s=>s.split(',').filter(Boolean),M:a=>[...new Set(a)],o:Boolean,eventToRollParams:()=>({skipDialog:true}),augmentInlineDamageRoll:async(formula,params)=>{augmented={formula,params};return {template:{damage:{roll:{kinds:new Set(['damage'])}}},context:{native:true}}},_loc:s=>s,DamagePF2e:{roll:async(template,context)=>{nativeRoll={template,context}}}};
 const Native=vm.runInNewContext('(class extends class {} {'+source.slice(start,end)+'})',context);await Native._onClickInlineRoll({target:anchor});
 assert.equal(augmented.formula,'4[electricity]');assert.equal(augmented.params.item,item);assert.equal(augmented.params.actor,origin);assert.equal(augmented.params.immutable,true);
 assert.ok(augmented.params.extraRollOptions.includes('pf2e-third-party-automation:metapower:chain-card:chain-use'));assert.ok(augmented.params.traits.includes('electricity'));assert.equal(nativeRoll.context.native,true);
});
