import test from 'node:test';
import assert from 'node:assert/strict';
let api={};try{api=await import('../scripts/manual-native-roll.mjs')}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error;}
function fixture(){
 const hooks=new Map();let sequence=0;const Hooks={on(event,fn){const id=++sequence;hooks.set(id,{event,fn});return id},off(_event,id){hooks.delete(id)},call(event,...args){let allowed=true;for(const hook of hooks.values())if(hook.event===event&&hook.fn(...args)===false)allowed=false;return allowed}};
 class DamageRoll {constructor(formula){this.formula='display without flavor';this.source=formula;this.options={rollerId:'owner',original:true};this._evaluated=false;}toJSON(){return {class:'DamageRoll',formula:this.source,total:this.total}}}
 const game={user:{id:'owner',isGM:false,settings:{showCheckDialogs:false,showDamageDialogs:false}},pf2e:{}};
 const anchors=[],document={createElement(tag){assert.equal(tag,'a');const element={dataset:{},parentElement:null};anchors.push(element);return element}};
 return {hooks,Hooks,DamageRoll,game,anchors,document};
}
for(const setting of [true,false])test(`manual events force a visible check and damage window with preference ${setting}`,()=>{
 assert.equal(typeof api.nativeRollEvent,'function');const game={user:{settings:{showCheckDialogs:setting,showDamageDialogs:setting}}};
 assert.deepEqual(api.nativeRollEvent(game),{shiftKey:!setting,ctrlKey:false,metaKey:false});assert.deepEqual(api.nativeRollEvent(game,'damage'),{shiftKey:!setting,ctrlKey:false,metaKey:false});assert.equal(game.user.settings.showCheckDialogs,setting);
});
for(const setting of [undefined,null])test(`an absent native preference ${setting} still forces its confirmation window`,()=>{const game={user:{settings:{showCheckDialogs:setting,showDamageDialogs:setting}}};assert.equal(api.nativeRollEvent(game).shiftKey,true);assert.equal(api.nativeRollEvent(game,'damage').shiftKey,true)});
test('damage window waits for native acceptance, captures only its marker, and never publishes a temporary card',async()=>{
 assert.equal(typeof api.manualDamageRoll,'function');const f=fixture(),original=globalThis.document;globalThis.document=f.document;let accept,entered;const entry=new Promise(r=>entered=r),choice=new Promise(r=>accept=r),input=new f.DamageRoll('{6d6[electricity]}');
 const unrelated={flags:{pf2e:{context:{type:'damage-roll',options:['other']}}},rolls:[{}]};
 const TextEditor={async _onClickInlineRoll(event){assert.equal(event.shiftKey,true);assert.equal(event.target.parentElement,null);assert.equal(event.target.dataset.baseFormula,'{6d6[electricity]}');assert.equal(event.target.dataset.immutable,'');assert.equal(event.target.dataset.overrideTraits,'');assert.equal(input._evaluated,false);assert.equal(f.Hooks.call('preCreateChatMessage',unrelated),true);entered();if(!await choice)return;const roll=new f.DamageRoll(event.target.dataset.baseFormula);roll._evaluated=true;roll.total=24;const message={author:{id:'owner'},flags:{pf2e:{context:{type:'damage-roll',options:[event.target.dataset.rollOptions]}}},rolls:[roll]};assert.equal(f.Hooks.call('preCreateChatMessage',message),false);}};
 try{let settled=false;const pending=api.manualDamageRoll({game:f.game,roll:input,Hooks:f.Hooks,TextEditor}).then(r=>{settled=true;return r});await entry;assert.equal(settled,false);assert.equal(input._evaluated,false);accept(true);const roll=await pending;assert.equal(roll.total,24);assert.equal(roll._evaluated,true);assert.equal(roll.options.original,true);assert.equal(f.hooks.size,0);assert.equal(input._evaluated,false);}finally{globalThis.document=original}
});
test('closing the native damage window returns null and removes its hook without evaluating or publishing',async()=>{
 assert.equal(typeof api.manualDamageRoll,'function');const f=fixture(),original=globalThis.document;globalThis.document=f.document;const roll=new f.DamageRoll('{2d8[healing]}');
 try{assert.equal(await api.manualDamageRoll({game:f.game,roll,Hooks:f.Hooks,TextEditor:{async _onClickInlineRoll(){}}}),null);assert.equal(roll._evaluated,false);assert.equal(f.hooks.size,0)}finally{globalThis.document=original}
});
test('native damage exceptions clean the hook and retain the original unevaluated roll',async()=>{
 assert.equal(typeof api.manualDamageRoll,'function');const f=fixture(),original=globalThis.document;globalThis.document=f.document;const roll=new f.DamageRoll('{1d8[slashing]}');
 try{await assert.rejects(api.manualDamageRoll({game:f.game,roll,Hooks:f.Hooks,TextEditor:{async _onClickInlineRoll(){throw Error('native failed')}}}),/native failed/);assert.equal(roll._evaluated,false);assert.equal(f.hooks.size,0)}finally{globalThis.document=original}
});
test('concurrent native windows capture their own damage only and leave unrelated native cards alone',async()=>{
 assert.equal(typeof api.manualDamageRoll,'function');const f=fixture(),original=globalThis.document;globalThis.document=f.document;const pending=[];
 const TextEditor={_onClickInlineRoll(event){return new Promise(resolve=>pending.push({event,resolve}))}};
 try{const a=api.manualDamageRoll({game:f.game,roll:new f.DamageRoll('{1d6[electricity]}'),Hooks:f.Hooks,TextEditor}),b=api.manualDamageRoll({game:f.game,roll:new f.DamageRoll('{2d8[healing]}'),Hooks:f.Hooks,TextEditor});await Promise.resolve();assert.equal(pending.length,2);assert.notEqual(pending[0].event.target.dataset.rollOptions,pending[1].event.target.dataset.rollOptions);
 for(const [i,p]of pending.entries()){const roll=new f.DamageRoll(p.event.target.dataset.baseFormula);roll._evaluated=true;roll.total=i?12:4;assert.equal(f.Hooks.call('preCreateChatMessage',{flags:{pf2e:{context:{type:'damage-roll',options:[p.event.target.dataset.rollOptions]}}},rolls:[roll],author:'owner',blind:true,whisper:['gm']}),false);p.resolve()}
 assert.equal((await a).total,4);assert.equal((await b).total,12);assert.equal(f.hooks.size,0)}finally{globalThis.document=original}
});
test('flat check confirmation uses the native Foundry dialog without rolling and accepts only its roll button',async()=>{
 assert.equal(typeof api.confirmManualFlatCheck,'function');let config;const Dialog={async wait(value){config=value;return true}};assert.equal(await api.confirmManualFlatCheck({label:'平检',dc:5,Dialog}),true);assert.equal(config.rejectClose,false);assert.equal(config.buttons[0].label,'投骰');assert.equal(config.buttons[1].label,'取消');assert.equal(await api.confirmManualFlatCheck({Dialog:{wait:async()=>null}}),false);
});
for(const [messageMode,blind,whisper]of [['blind',true,['gm']],['gm',false,['gm']],['self',false,['owner']],['public',false,[]]])test(`manual damage retains the actual native ${messageMode} audience for final publication`,async()=>{
 assert.equal(typeof api.manualDamagePrivacy,'function');const f=fixture(),original=globalThis.document;globalThis.document=f.document;
 try{const result=await api.manualDamageRoll({game:f.game,roll:new f.DamageRoll('{2d6[electricity]}'),Hooks:f.Hooks,TextEditor:{async _onClickInlineRoll(event){const roll=new f.DamageRoll(event.target.dataset.baseFormula);roll._evaluated=true;roll.total=6;assert.equal(f.Hooks.call('preCreateChatMessage',{author:'owner',blind,whisper,flags:{pf2e:{context:{type:'damage-roll',messageMode,options:[event.target.dataset.rollOptions]}}},rolls:[roll]}),false)}}});const privacy=api.manualDamagePrivacy(result);assert.deepEqual(privacy,{messageMode,blind,whisper});privacy.whisper.push('other');assert.deepEqual(api.manualDamagePrivacy(result).whisper,whisper);assert.equal(f.hooks.size,0)}finally{globalThis.document=original}
});
test('a nonce copied onto another author cannot become this native roll',async()=>{
 const f=fixture(),original=globalThis.document;globalThis.document=f.document;
 try{await assert.rejects(api.manualDamageRoll({game:f.game,roll:new f.DamageRoll('{1d6[electricity]}'),Hooks:f.Hooks,TextEditor:{async _onClickInlineRoll(event){const roll=new f.DamageRoll(event.target.dataset.baseFormula);roll._evaluated=true;roll.total=3;assert.equal(f.Hooks.call('preCreateChatMessage',{author:'other',flags:{pf2e:{context:{type:'damage-roll',options:[event.target.dataset.rollOptions]}}},rolls:[roll]}),false)}}}),/原生伤害/);assert.equal(f.hooks.size,0)}finally{globalThis.document=original}
});
test('a fixed audience is visible and locked in its own native select and cannot be changed before publication',async()=>{
 const f=fixture(),original=globalThis.document;globalThis.document=f.document;const select={value:'public',disabled:false};
 try{const result=await api.manualDamageRoll({game:f.game,roll:new f.DamageRoll('{2d8[healing]}'),messageMode:'blind',Hooks:f.Hooks,TextEditor:{async _onClickInlineRoll(event){const app={context:{options:new Set([event.target.dataset.rollOptions]),messageMode:'public'}};f.Hooks.call('renderDamageModifierDialog',app,[{querySelector:selector=>selector==='select[name=messageMode]'?select:null}]);assert.equal(app.context.messageMode,'blind');assert.equal(select.value,'blind');assert.equal(select.disabled,true);const roll=new f.DamageRoll(event.target.dataset.baseFormula);roll._evaluated=true;roll.total=8;f.Hooks.call('preCreateChatMessage',{author:'owner',blind:true,whisper:['gm'],flags:{pf2e:{context:{type:'damage-roll',options:[event.target.dataset.rollOptions],messageMode:app.context.messageMode}}},rolls:[roll]})}}});assert.equal(api.manualDamagePrivacy(result).blind,true);assert.equal(f.hooks.size,0)}finally{globalThis.document=original}
});
test('the original private audience cannot be widened by a public native choice',()=>{
 const roll={};assert.deepEqual(api.manualDamagePrivacy(roll,{messageMode:'blind',blind:true,whisper:['gm']}),{messageMode:'blind',blind:true,whisper:['gm']});assert.throws(()=>api.manualDamagePrivacy(roll,{blind:true,whisper:[]}),/受众/);
});
test('a source self audience keeps its recipient instead of using the later executor self mode',()=>{
 assert.deepEqual(api.manualDamagePrivacy({}, {messageMode:'self',blind:false,whisper:['original-owner']}),{messageMode:'gm',blind:false,whisper:['original-owner']});
});

test('an ordinary HTTP client uses Foundry randomID without requiring secure-context crypto.randomUUID',async()=>{
 const f=fixture(),original=globalThis.document,foundry=globalThis.foundry,descriptor=Object.getOwnPropertyDescriptor(globalThis,'crypto');globalThis.document=f.document;globalThis.foundry={utils:{randomID:()=> 'native-id'}};Object.defineProperty(globalThis,'crypto',{configurable:true,value:{}});
 try{assert.equal(await api.manualDamageRoll({game:f.game,roll:new f.DamageRoll('{1d6[electricity]}'),Hooks:f.Hooks,TextEditor:{async _onClickInlineRoll(event){assert.equal(event.target.dataset.rollOptions,'pf2e-third-party-automation:manual-damage:native-id')}}}),null);assert.equal(f.hooks.size,0)}finally{globalThis.document=original;globalThis.foundry=foundry;Object.defineProperty(globalThis,'crypto',descriptor)}
});

test('a guarded native damage window closes before dice after its original owner disconnects',{timeout:250},async()=>{
 const f=fixture(),original=globalThis.document;globalThis.document=f.document;let active=true,app,evaluations=0;const assertLive=()=>{if(!active)throw Error('owner disconnected')};
 const TextEditor={async _onClickInlineRoll(event){let resolve;const choice=new Promise(r=>resolve=r);app={context:{options:[event.target.dataset.rollOptions]},resolve,close(){this.resolve(false)}};f.Hooks.call('renderDamageModifierDialog',app);if(await choice)evaluations++}};
 try{const pending=api.manualDamageRoll({game:f.game,roll:new f.DamageRoll('{1d6[electricity]}'),Hooks:f.Hooks,TextEditor,assertLive}).then(value=>({value}),error=>({error}));await new Promise(resolve=>setImmediate(resolve));active=false;f.Hooks.call('userConnected',f.game.user,false);const result=await pending;assert.match(result.error?.message??'',/disconnected/);await app.resolve(true);assert.equal(evaluations,0);assert.equal(f.hooks.size,0)}finally{globalThis.document=original}
});

test('a guarded damage operation stays aborted when its native dialog finishes preparing after role recovery',{timeout:250},async()=>{
 const f=fixture(),original=globalThis.document;globalThis.document=f.document;let alive=true,prepare,app,evaluations=0;const prepared=new Promise(resolve=>prepare=resolve),assertLive=()=>{if(!alive)throw Error('original GM changed')};
 const TextEditor={async _onClickInlineRoll(event){await prepared;let resolve;const choice=new Promise(r=>resolve=r);app={context:{options:[event.target.dataset.rollOptions]},resolve,close(){this.resolve(false)}};f.Hooks.call('renderDamageModifierDialog',app);if(await choice)evaluations++}};
 try{const pending=api.manualDamageRoll({game:f.game,roll:new f.DamageRoll('{1d6[electricity]}'),Hooks:f.Hooks,TextEditor,assertLive}).then(value=>({value}),error=>({error}));await new Promise(resolve=>setImmediate(resolve));alive=false;f.Hooks.call('userConnected',f.game.user,false);const result=await pending;assert.match(result.error?.message??'',/GM changed/);alive=true;prepare();await new Promise(resolve=>setImmediate(resolve));await app.resolve(true);await new Promise(resolve=>setImmediate(resolve));assert.equal(evaluations,0);assert.equal(f.hooks.size,0)}finally{globalThis.document=original}
});
