import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {installActionEntrances} from '../scripts/metapower/entrances.mjs';
let api={};try{api=await import('../scripts/native-action-events.mjs')}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error}

function fixture(){
 const calls=[],actor={uuid:'Actor.a'},other={uuid:'Actor.b'};
 class Variant{constructor(action){this.action=action;this.slug=action.slug}async use(params){calls.push({variant:this,params});return params.result??'native result'}async toMessage(){return 'display'}}
 const action={slug:'sustain',variants:new Map(),toActionVariant(){return new Variant(this)},async use(params){return this.toActionVariant().use(params)}};
 const cached=action.toActionVariant();action.variants.set('cached',cached);
 const game={user:{id:'user',getActiveTokens:()=>[{actor}],character:other},pf2e:{actions:new Map([['sustain',action]])}};
 return {game,action,cached,actor,other,calls,Variant};
}
function events(f){assert.equal(typeof api.getNativeActionEvents,'function');return api.getNativeActionEvents({game:f.game})}

test('one shared observer gives two subscribers the same branded frozen scope around one original use',async t=>{
 const f=fixture(),e=events(f),seen=[],order=[];t.after(()=>e.cleanup());
 assert.equal(api.getNativeActionEvents({game:f.game}),e);
 e.addMiddleware(async(s,next)=>{seen.push(s);order.push('first');const result=await next();order.push('last');return result});
 e.addMiddleware(async(s,next)=>{seen.push(s);order.push('second');return next()});e.register();e.register();
 const params={actors:[f.actor,f.actor,f.other],message:{create:true},result:{message:'exact'}};
 assert.equal(await f.action.use(params),params.result);assert.equal(f.calls.length,1);assert.equal(f.calls[0].params,params);
 assert.deepEqual(order,['first','second','last']);assert.equal(seen[0],seen[1]);assert.equal(seen[0].action,f.action);assert.equal(seen[0].variant,f.calls[0].variant);assert.equal(seen[0].slug,'sustain');assert.equal(seen[0].user,f.game.user);
 assert.deepEqual(seen[0].actors,[f.actor,f.other]);assert.equal(Object.isFrozen(seen[0]),true);assert.equal(Object.isFrozen(seen[0].actors),true);assert.equal(Object.isFrozen(seen[0].params),true);assert.equal(Object.isFrozen(seen[0].params.actors),true);assert.equal(Object.isFrozen(seen[0].params.message),true);assert.equal(Object.isFrozen(f.actor),false);
 params.actors.length=0;params.message.create=false;assert.deepEqual(seen[0].actors,[f.actor,f.other]);assert.equal(seen[0].params.message.create,true);
});

test('cached and future variants are branded; copied methods, spoofed factory receiver and displays are not',async t=>{
 const f=fixture(),e=events(f),seen=[];e.addMiddleware((s,next)=>{seen.push(s);return next()});e.register();t.after(()=>e.cleanup());
 await f.cached.use({});await f.action.toActionVariant().use({});assert.equal(seen.length,2);
 const copied=new f.Variant(f.action);copied.use=f.cached.use;await copied.use({});
 await f.action.toActionVariant.call({...f.action}).use({});await f.action.toActionVariant().toMessage();assert.equal(seen.length,2);
});

test('draft use bypasses middleware but preserves original side effects and return',async t=>{
 const f=fixture(),e=events(f);e.addMiddleware(()=>{throw Error('not an actual use')});e.register();t.after(()=>e.cleanup());
 assert.equal(await f.action.use({message:{create:false}}),'native result');assert.equal(f.calls.length,1);
});

test('implicit actor selection and assigned fallback are frozen before middleware awaits',async t=>{
 const f=fixture(),e=events(f),seen=[];e.addMiddleware(async(s,next)=>{seen.push(s);f.game.user.getActiveTokens=()=>[];await Promise.resolve();return next()});e.register();t.after(()=>e.cleanup());
 await f.action.use({});await f.action.use({});assert.deepEqual(seen.map(s=>s.actors),[[f.actor],[f.other]]);
});

test('veto and throw do not enter native; original native error retains identity',async t=>{
 const f=fixture(),e=events(f),error=Error('native failure');e.register();t.after(()=>e.cleanup());
 let remove=e.addMiddleware(()=>false);assert.equal(await f.cached.use({}),false);assert.equal(f.calls.length,0);remove();
 remove=e.addMiddleware(()=>{throw error});e.register();await assert.rejects(f.cached.use({}),e=>e===error);assert.equal(f.calls.length,0);remove();
 const action={slug:'other',variants:[],toActionVariant:()=>({use:async()=>{throw error}})};f.game.pf2e.actions.set('other',action);e.addMiddleware((_s,next)=>next());e.register();await assert.rejects(action.toActionVariant().use({}),e=>e===error);
});

test('a middleware cannot replay the same native action or retain a live continuation after return',async t=>{
 const f=fixture(),e=events(f);let late;e.addMiddleware(async(_s,next)=>{late=next;const result=await next();await assert.rejects(next(),/once|closed/i);return result});e.register();t.after(()=>e.cleanup());
 assert.equal(await f.cached.use({}),'native result');await assert.rejects(late(),/once|closed/i);assert.equal(f.calls.length,1);
});

test('unsubscribing one consumer keeps another; last unsubscribe restores original factory and cached use',async()=>{
 const f=fixture(),originalFactory=f.action.toActionVariant,originalUse=f.cached.use,e=events(f),seen=[];
 const first=e.addMiddleware((_s,next)=>{seen.push('first');return next()}),second=e.addMiddleware((_s,next)=>{seen.push('second');return next()});e.register();first();first();await f.cached.use({});assert.deepEqual(seen,['second']);
 second();assert.equal(f.action.toActionVariant,originalFactory);assert.equal(f.cached.use,originalUse);assert.equal(Object.hasOwn(f.cached,'use'),false);
 e.addMiddleware((_s,next)=>{seen.push('new');return next()});e.register();await f.cached.use({});assert.deepEqual(seen,['second','new']);e.cleanup();
});

test('complete subclass use including super and post-effects produces one observer result',async t=>{
 const f=fixture();class Child extends f.Variant{async use(params){const value=await super.use(params);f.calls.push('post-effect');return `${value}:child`}}
 const action={slug:'raise-a-shield',variants:[],toActionVariant:()=>new Child(action)};f.game.pf2e.actions.set(action.slug,action);
 const e=events(f),seen=[];e.addMiddleware(async(s,next)=>{const value=await next();seen.push([s.action,value,f.calls.at(-1)]);return value});e.register();t.after(()=>e.cleanup());
 assert.equal(await action.toActionVariant().use({}),'native result:child');assert.deepEqual(seen,[[action,'native result:child','post-effect']]);
});

for(const order of ['metapower-first','observer-first'])test(`cached/future native uses preserve the real metapower wrapper chain (${order})`,async()=>{
 const f=fixture(),e=events(f),log=[];e.addMiddleware(async(_s,next)=>{log.push('action');const result=await next();log.push('action-end');return result});
 const install=()=>installActionEntrances({game:f.game,eligible:()=>true,observe:async({actor},next)=>{log.push(`meta:${actor.uuid}`);const result=await next();log.push('meta-end');return result}});
 let remove;if(order==='metapower-first'){remove=install();e.register()}else{e.register();remove=install()}
 for(const variant of [f.cached,f.action.toActionVariant()]){log.length=0;assert.equal(await variant.use({actors:[f.actor]}),'native result');assert.equal(log.filter(x=>x==='action').length,1);assert.equal(log.filter(x=>x==='meta:Actor.a').length,1);assert.equal(log.filter(x=>x==='action-end').length,1);assert.equal(log.filter(x=>x==='meta-end').length,1)}
 e.cleanup();log.length=0;await f.action.toActionVariant().use({actors:[f.actor]});assert.deepEqual(log,['meta:Actor.a','meta-end']);remove();
});

const nativePath=process.env.PF2E_NATIVE_BUNDLE??'';
test('actual PF2e SimpleAction Sustain factory and returned actor/message rows preserve native create/veto semantics',{skip:!nativePath},async t=>{
 const source=readFileSync(nativePath,'utf8'),start=source.indexOf('var BaseActionVariant = class {'),end=source.indexOf('function toRollNoteSource',start);assert.ok(start>=0&&end>start);
 const actor={uuid:'Actor.native',isOwner:true},messages=[];let veto=false;
 class NativeMessage{constructor(data){Object.assign(this,data)}static async create(data){if(veto)return undefined;const m=new NativeMessage(data);m.id=`message${messages.length}`;messages.push(m);return m}}
 const CONFIG={PF2E:{actionTraits:{concentrate:'Concentrate'},traitsDescriptions:{}}},foundry={applications:{handlebars:{renderTemplate:async()=>'<native flavor>'}}};
 const SimpleAction=new Function('Collection','sluggify','getActionGlyph','_loc','CONFIG','foundry','ChatMessagePF2e','ChatMessage','getSelectedActors',`${source.slice(start,end)};return SimpleAction;`)(Map,s=>s.toLowerCase().replaceAll(' ','-'),()=>1,s=>s,CONFIG,foundry,NativeMessage,{getSpeaker:({actor})=>({actor:actor.uuid})},()=>[actor]);
 const sustainSource=source.match(/hm = new SimpleAction\(\{[\s\S]*?slug: `sustain`,[\s\S]*?\}\)/)?.[0];assert.ok(sustainSource);
 const sustain=new Function('SimpleAction',`return ${sustainSource.slice('hm = '.length)}`)(SimpleAction),game={pf2e:{actions:new Map([['sustain',sustain]])},user:{id:'native',getActiveTokens:()=>[{actor}]}};
 const e=events({game}),seen=[];e.addMiddleware(async(scope,next)=>{const result=await next();seen.push({scope,result});return result});e.register();t.after(()=>e.cleanup());
 const result=await sustain.use({actors:[actor]});assert.equal(result[0].actor,actor);assert.equal(result[0].message,messages[0]);assert.equal(seen[0].result,result);assert.equal(seen[0].scope.action,sustain);assert.equal(seen[0].scope.variant.slug,'sustain');assert.equal(seen[0].scope.params.actors[0],actor);
 await sustain.toMessage();assert.equal(seen.length,1);await sustain.use({message:{create:false}});assert.equal(seen.length,1);
 veto=true;const cancelled=await sustain.use({actors:[actor]});assert.equal(cancelled[0].message,undefined);assert.equal(seen.length,2);assert.equal(seen[1].result,cancelled);assert.equal('completed' in seen[1].scope,false);
});
