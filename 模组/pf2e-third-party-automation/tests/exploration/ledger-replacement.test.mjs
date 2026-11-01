// A ledger only grows, and sessions and clocks an earlier version left open keep blocking new recoveries.
// Replacing it starts an empty ledger beside the old one, which is never written again and stays as the archive.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createDocumentStore} from '../../scripts/exploration/document-store.mjs';
import {createLedgerSetup} from '../../scripts/exploration/setup.mjs';
import {createLedger} from '../../scripts/exploration/ledger.mjs';
import {createCoordinator} from '../../scripts/exploration/coordinator.mjs';
import {createRecoveryPanel} from '../../scripts/exploration/panel.mjs';
import {createExplorationRuntime} from '../../scripts/exploration/runtime.mjs';
import {chooseNext} from '../../scripts/exploration/policy.mjs';
import {createGenesisRevision,encodeRevision,decodeRevision,revisionId} from '../../scripts/exploration/revision-codec.mjs';
import {MODULE_ID,emptyLedger} from '../../scripts/exploration/schema.mjs';

const confirmations={issuersStopped:true,clientsReloaded:true,recoveryDisabled:true};
const OLD='JournalEntry.OLDROOT000000001',NEW='JournalEntry.NEWROOT000000001',OTHER='JournalEntry.OTHER00000000001',H='Actor.H',P='Actor.P';

// One world's ledger as earlier versions left it: four stopped automatic sessions, the last with a time effect that was
// never confirmed, and a stopped manual session. A finished session and an accepted one ride along; neither is left open.
function stoppedLedger(){
 const session=(id,startedAt,patch={})=>({id,actorUUIDs:[H,P],activityIds:[],startedAt,cursorAt:startedAt+600,budgetEndsAt:startedAt+7200,maxActivities:100,goalsByPool:[{poolUUID:H,targetHP:20},{poolUUID:P,targetHP:20}],manual:false,protocol:{version:1,rootUUID:OLD,epoch:'old-epoch'},status:'paused',stopReason:'user-stopped',...patch});
 const clock=(id,sessionId,from,patch={})=>({id,sessionId,gmId:'G',from,to:from+600,state:'confirmed',evidence:[],...patch});
 const seed=emptyLedger();
 for(const [i,[id,stopReason]] of [['A','blocked'],['B','encounter-started'],['C','world-time-conflict'],['D','encounter-started']].entries()){seed.sessions[id]=session(id,i*1800,{stopReason});seed.clocks[`clock-${id}`]=clock(`clock-${id}`,id,i*1800)}
 seed.clocks.unsettled=clock('unsettled','D',6000,{state:'uncertain',reason:'time-effects-unconfirmed'});
 seed.sessions.M=session('M',9000,{manual:true,cursorAt:9000,manualPoolIssuer:{userId:'G',clientNonce:'earlier-tab'}});
 seed.sessions.E=session('E',8000,{status:'complete',stopReason:'goals-met'});
 seed.sessions.F=session('F',7200,{status:'closed',stopReason:'accepted-current-state',recoveryReview:{disposition:'accepted-current-state',userId:'G',at:7800,clockIds:['accepted'],note:'checked'}});
 seed.clocks.accepted=clock('accepted','F',7200,{state:'uncertain',reason:'time-effects-unconfirmed',nativeResolved:true,evidence:[{userId:'G',worldTime:7800,dt:600,options:{pf2eThirdPartyAutomation:{exploration:{checkpointId:'accepted',sessionId:'F',gmId:'G',expectedFrom:7200,expectedTo:7800}}}}]});
 return seed;
}

// The server side of one world: every journal root by UUID, the ledger setting, and this GM's user flags.
// A reload is a new store or runtime on the same game.
async function world({seed,initialized=true,configured=OLD}={}){
 const docs=new Map(),requests=[],settingsWrites=[],events=[],flags={};
 let setting=configured,created=0,sequence=0,alterACK=null,settingHook=null,hooks=new Map();
 if(seed){
  const raw={_id:OLD.split('.')[1],name:'探索活动账本',ownership:{default:0,G:3},flags:{[MODULE_ID]:{explorationLedger:structuredClone(seed)}},pages:[]};
  if(initialized){const genesis=await createGenesisRevision({rootUUID:OLD,epoch:'old-epoch',nonce:'old-genesis',writerUserId:'G',writerClientId:'earlier-tab',seed});raw.pages.push({_id:revisionId(0),name:'Exploration revision 0',type:'text',ownership:{default:0},flags:{[MODULE_ID]:{explorationRevision:encodeRevision(genesis)}}})}
  docs.set(OLD,raw);
 }
 // Foundry runs updateUser on the client that changed the user as well, so clearing the flag reaches this page's own hook.
 const gm={id:'G',isGM:true,active:true,getFlag:(_scope,key)=>flags[key]??null,setFlag:async(_scope,key,value)=>{flags[key]=value},unsetFlag:async(_scope,key)=>{delete flags[key];for(const handler of hooks.get('updateUser')??[])handler(gm)}};
 const users=new Map([['G',gm],['G2',{id:'G2',isGM:true,active:true}]]);users.activeGM=gm;
 const actors=new Map([['H',20],['P',1]].map(([id,hp])=>[id,{id,uuid:`Actor.${id}`,name:id,type:'character',items:[],testUserPermission:()=>true,system:{attributes:{hp:{value:hp,max:20,temp:0}},resources:{focus:{value:0,max:0}}}}]));
 const game={user:gm,users,actors,modules:new Map(),packs:new Map(),messages:new Map(),time:{worldTime:9000},system:{id:'pf2e',version:'8.6.1'},pf2e:{actions:new Map()},paused:false,togglePause:paused=>{game.paused=paused;return paused},
  settings:{get:(_scope,key)=>key==='explorationLedgerUUID'?setting:undefined,set:async(scope,key,value)=>{
   settingsWrites.push(value);if(settingHook)return settingHook(value);
   setting=value;events.push(`setting ${value}`);for(const handler of hooks.get('updateSetting')??[])handler({key:`${scope}.${key}`});return value;
  }},
  socket:{id:'socket',connected:true,on(){},off(){},emit(event,request,respond){
   assert.equal(event,'modifyDocument');requests.push(structuredClone(request));const {type,action,operation}=request,userId=game.user.id;let result=[],error;
   if(type==='JournalEntry'&&action==='get'){const found=docs.get(`JournalEntry.${operation.query._id}`);if(found)result=[structuredClone(found)]}
   else if(type==='JournalEntry'&&action==='create'){
    const data=structuredClone(operation.data[0]);data._id=`NEWROOT${String(++created).padStart(9,'0')}`;data.ownership[userId]=3;data.pages=[];
    docs.set(`JournalEntry.${data._id}`,data);events.push(`root JournalEntry.${data._id}`);result=[structuredClone(data)];
   }else if(type==='JournalEntryPage'&&action==='create'){
    const root=docs.get(operation.parentUuid),page=structuredClone(operation.data[0]);assert.ok(root,'pages are only created under an existing root');
    if(root.pages.some(saved=>saved._id===page._id))error={class:'ServerError',message:`The _id [${page._id}] already exists within the parent collection: JournalEntry [${root._id}] pages`};
    else{root.pages.push(page);events.push(`page ${operation.parentUuid}`);result=[structuredClone(page)]}
   }else assert.fail(`unexpected ${type} ${action}`);
   const ack={type,action,operation:structuredClone(operation),broadcast:false,userId,result,...error?{error}:{}};
   queueMicrotask(()=>alterACK?alterACK(ack,respond,request):respond(ack));
  }}};
 const store=name=>createDocumentStore({game,writerClientId:name,nonce:()=>`nonce-${++sequence}`,timeoutMs:500,rootStamp:()=>'2026-10-13 21:07'});
 function client(name){const tab=store(name),ledger=createLedger({...tab,isAuthority:()=>true,identity:()=>({userId:game.user.id,clientNonce:name})});return {store:tab,ledger,setup:createLedgerSetup({game,store:tab,ledger,writerClientId:name})}}
 function runtime(){
  hooks=new Map();const errors=[],nativeCasts={addMatcher(){},addCapture(){},addActorUpdateMiddleware(){},addConsumePolicy(){},addObserver(){}};
  const Hooks={on(event,handler){hooks.set(event,[...hooks.get(event)??[],handler]);return handler},off(){}};
  const page=hooks,emitHook=(event,...args)=>{for(const handler of page.get(event)??[])handler(...args)};
  return {errors,emitHook,...createExplorationRuntime({game,Hooks,fromUuid:async uuid=>actors.get(uuid.split('.')[1])??null,nativeCasts,onError:error=>errors.push(error)})};
 }
 return {game,docs,requests,settingsWrites,events,flags,store,client,runtime,bytes:uuid=>JSON.stringify(docs.get(uuid)),setting:()=>setting,setSetting:value=>{setting=value},setACK:fn=>{alterACK=fn},setSettingHook:fn=>{settingHook=fn}};
}
const writes=w=>w.requests.filter(request=>request.action!=='get').map(request=>[request.type,request.operation.parentUuid??null]);

// Foundry's dialog, notifications and form reading, with the dialog answered by pressing one of its buttons.
const natives={foundry:globalThis.foundry,ui:globalThis.ui,FormData:globalThis.FormData};
function stubFoundry(t,answer=()=>null){
 const dialogs=[],infos=[];
 t.after(()=>Object.assign(globalThis,natives));
 globalThis.FormData=class{constructor(form){this.form=form}has(key){return this.form[key]===true}};
 globalThis.ui={notifications:{info:message=>infos.push(message)}};
 globalThis.foundry={applications:{api:{ApplicationV2:class{render(){return this}},DialogV2:{wait:async options=>{dialogs.push(options);return answer(options)}}}}};
 return {dialogs,infos};
}
const press=(action,form={})=>options=>options.buttons.find(button=>button.action===action).callback(null,{form});
const actionsIn=html=>[...html.matchAll(/data-recovery="([^"]+)"/g)].map(match=>match[1]);
const mainActions=html=>actionsIn(html.match(/<div class="recovery-actions recovery-main-actions">([\s\S]*?)<\/div>/)[1]);
const moreActions=html=>html.match(/<details class="recovery-more"><summary>更多操作<\/summary><div class="recovery-actions">([\s\S]*?)<\/div>/)[1];

test('replacement creates an initialized empty root, then moves the setting and retires this runtime',async()=>{
 const w=await world({seed:stoppedLedger()}),store=w.store('this-tab');await store.read();
 assert.deepEqual(await store.replace(confirmations),{rootUUID:NEW,previousRootUUID:OLD,requiresReload:true,initialized:true});
 assert.equal(w.setting(),NEW);assert.deepEqual(w.settingsWrites,[NEW]);
 // Genesis is on the new root before the setting names it, so no runtime ever opens that root as legacy data.
 assert.deepEqual(w.events,[`root ${NEW}`,`page ${NEW}`,`setting ${NEW}`]);
 const root=w.docs.get(NEW);assert.equal(root.ownership.default,0);
 // The archive keeps its name; the new root is told apart by the day it was made.
 assert.equal(root.name,'探索活动账本 2026-10-13 21:07');assert.equal(w.docs.get(OLD).name,'探索活动账本');
assert.deepEqual(root.pages.map(page=>page._id),[revisionId(0)]);
 const genesis=decodeRevision(root.pages[0].flags[MODULE_ID].explorationRevision);assert.equal(genesis.rootUUID,NEW);assert.equal(genesis.revision,0);assert.deepEqual(genesis.seed,emptyLedger());
 assert.deepEqual(root.flags[MODULE_ID].explorationLedger,emptyLedger());assert.equal(root.flags[MODULE_ID].explorationLedgerRoot.schemaVersion,1);
 for(const operation of [()=>store.read(),()=>store.status(),()=>store.inspect(),()=>store.transact(()=>{})])await assert.rejects(operation(),/revision-root-changed/);
 const reloaded=w.store('reloaded-tab'),status=await reloaded.status();
 assert.deepEqual(status,{initialized:true,rootUUID:NEW,epoch:genesis.epoch,revision:0,sourceDigest:genesis.sourceDigest});assert.deepEqual(await reloaded.read(),emptyLedger());
});

for(const initialized of [true,false])test(`the replaced ${initialized?'initialized':'uninitialized'} root receives no page and no flag change`,async()=>{
 const w=await world({seed:initialized?stoppedLedger():emptyLedger(),initialized}),before=w.bytes(OLD);
 await w.store('this-tab').replace(confirmations);
 assert.equal(w.bytes(OLD),before);assert.deepEqual(writes(w),[['JournalEntry',null],['JournalEntryPage',NEW]]);
});

test('the storage layer refuses replacement without every confirmation, without a configured ledger, after a switch, and from a GM who is not active',async()=>{
 for(const key of Object.keys(confirmations)){const w=await world({seed:stoppedLedger()});await assert.rejects(w.store('this-tab').replace({...confirmations,[key]:false}),/setup-confirmations-required/);assert.deepEqual(w.requests,[]);assert.deepEqual(w.settingsWrites,[])}
 const unconfigured=await world({configured:''});await assert.rejects(unconfigured.store('this-tab').replace(confirmations),/root-not-configured/);assert.deepEqual(unconfigured.requests,[]);
 const inactive=await world({seed:stoppedLedger()});inactive.game.users.activeGM=inactive.game.users.get('G2');await assert.rejects(inactive.store('this-tab').replace(confirmations),/active-gm-required/);assert.deepEqual(inactive.requests,[]);
 const switched=await world({seed:stoppedLedger()}),store=switched.store('this-tab');await store.replace(confirmations);const sent=switched.requests.length;
 await assert.rejects(store.replace(confirmations),/root-runtime-reload-required/);assert.equal(switched.requests.length,sent);assert.deepEqual(switched.settingsWrites,[NEW]);
});

const afterCreation={
 'an unknown genesis acknowledgement':{arm:w=>w.setACK((ack,respond,request)=>{if(request.type==='JournalEntryPage')ack.result=[];respond(ack)}),error:/revision-acknowledgement-unknown/},
 'a genesis page the server changed':{arm:w=>w.setACK((ack,respond,request)=>{if(request.type==='JournalEntryPage')ack.result[0].name='renamed';respond(ack)}),error:/revision-acknowledgement-mismatch/},
 'authority lost while genesis is written':{arm:w=>w.setACK((ack,respond,request)=>{if(request.type==='JournalEntryPage')w.game.users.activeGM=w.game.users.get('G2');respond(ack)}),error:/active-gm-required/},
 'a competing setting change':{arm:w=>w.setACK((ack,respond,request)=>{if(request.type==='JournalEntryPage')w.setSetting(OTHER);respond(ack)}),error:/root-config-conflict/,setting:OTHER},
 'a rejected setting write':{arm:w=>w.setSettingHook(()=>{throw Error('setting-outcome-unknown')}),error:/setting-outcome-unknown/,attempted:[NEW]}
};
for(const [name,{arm,error,setting=OLD,attempted=[]}] of Object.entries(afterCreation))test(`${name} keeps the setting off the new root and names the orphan`,async()=>{
 const w=await world({seed:stoppedLedger()}),before=w.bytes(OLD);arm(w);
 await assert.rejects(w.store('this-tab').replace(confirmations),caught=>{assert.match(caught.message,error);assert.equal(caught.rootUUID,NEW);return true});
 assert.equal(w.setting(),setting);assert.deepEqual(w.settingsWrites,attempted);assert.equal(w.bytes(OLD),before);assert.ok(w.docs.has(NEW));
});

test('a root whose creation cannot be verified is never initialized or configured, and the same page can try again',async()=>{
 const w=await world({seed:stoppedLedger()}),store=w.store('this-tab'),before=w.bytes(OLD);
 w.setACK((ack,respond,request)=>{if(request.type==='JournalEntry'&&request.action==='create')ack.result[0].flags[MODULE_ID].explorationLedgerRoot.setupNonce='unrelated';respond(ack)});
 await assert.rejects(store.replace(confirmations),/root-create-acknowledgement-mismatch/);
 assert.deepEqual(writes(w),[['JournalEntry',null]]);assert.equal(w.setting(),OLD);assert.deepEqual(w.settingsWrites,[]);
 w.setACK(null);const second='JournalEntry.NEWROOT000000002';
 assert.equal((await store.replace(confirmations)).rootUUID,second);assert.equal(w.setting(),second);assert.equal(w.docs.get(NEW).pages.length,0);assert.equal(w.bytes(OLD),before);
});

for(const [status,id,patch] of [['running','D',{driver:{userId:'G',clientNonce:'earlier-tab',leaseNonce:'lease'}}],['recording','M',{}]])test(`a ${status} session refuses replacement before anything is created`,async()=>{
 const seed=stoppedLedger();Object.assign(seed.sessions[id],{status,stopReason:null,...patch});
 const w=await world({seed}),{setup}=w.client('this-tab');assert.equal((await setup.status()).state,'ready');
 await assert.rejects(setup.replacePreview(),/replace-ledger-session-active/);await assert.rejects(setup.replace(confirmations),/replace-ledger-session-active/);
 assert.deepEqual(writes(w),[]);assert.equal(w.setting(),OLD);assert.deepEqual(w.settingsWrites,[]);
});

test('stopped sessions and an unconfirmed clock are counted, then left behind in the archive',async()=>{
 const w=await world({seed:stoppedLedger()}),{setup}=w.client('this-tab'),before=w.bytes(OLD);
 // A, B, C, D and M never finished; E is complete and F closed. The clock F's review accepted no longer blocks, D's does.
 assert.deepEqual(await setup.replacePreview(),{unfinishedSessions:5,unconfirmedClocks:1});assert.deepEqual(writes(w),[]);
 assert.deepEqual(await setup.replace(confirmations),{rootUUID:NEW,previousRootUUID:OLD,requiresReload:true,initialized:true});assert.equal(w.setting(),NEW);assert.equal(w.bytes(OLD),before);
 const reloaded=w.client('reloaded-tab');assert.equal((await reloaded.setup.status()).state,'ready');assert.deepEqual(await reloaded.store.read(),emptyLedger());
 assert.deepEqual(await reloaded.setup.replacePreview(),{unfinishedSessions:0,unconfirmedClocks:0});
});

// The README tells a GM who deleted the live ledger instead of the archive how to get back.
test('after the live ledger is deleted by mistake, the archive can be selected again and replaced once more',async()=>{
 const w=await world({seed:stoppedLedger()});await w.client('this-tab').setup.replace(confirmations);w.docs.delete(NEW);
 const lost=w.client('reloaded-tab');assert.deepEqual(await lost.setup.status(),{state:'blocked',reason:'root-not-found'});
 await assert.rejects(lost.setup.replace(confirmations),/root-not-found/);
 assert.equal((await lost.setup.select({...confirmations,rootUUID:OLD})).requiresReload,true);assert.equal(w.setting(),OLD);
 const back=w.client('second-reload');assert.equal((await back.setup.status()).state,'ready');
 const again=await back.setup.replace(confirmations);assert.equal(again.previousRootUUID,OLD);assert.notEqual(again.rootUUID,NEW);assert.equal(w.setting(),again.rootUUID);
 const status=await w.client('third-reload').setup.status();assert.equal(status.state,'ready');assert.equal(status.revision,0);
});
test('only an active GM with every confirmation can replace, and only a ready ledger',async()=>{
 const refused=async(w,error)=>{const {setup}=w.client('this-tab');await assert.rejects(setup.replace(confirmations),error);await assert.rejects(setup.replacePreview(),error);assert.deepEqual(writes(w),[]);assert.deepEqual(w.settingsWrites,[])};
 const legacy=await world({seed:stoppedLedger(),initialized:false});assert.equal((await legacy.client('status').setup.status()).state,'legacy');await refused(legacy,/replace-ledger-ready-required/);
 const unmigrated=stoppedLedger(),{protocol,...withoutProtocol}=unmigrated.sessions.D;unmigrated.sessions.D={...withoutProtocol,status:'running'};
 const migrating=await world({seed:unmigrated});assert.equal((await migrating.client('status').setup.status()).state,'migration-required');await refused(migrating,/replace-ledger-ready-required/);
 await refused(await world({configured:''}),/root-not-configured/);
 const inactive=await world({seed:stoppedLedger()});inactive.game.users.activeGM=inactive.game.users.get('G2');await refused(inactive,/active-gm-required/);assert.deepEqual(inactive.requests,[]);
 for(const key of Object.keys(confirmations)){const w=await world({seed:stoppedLedger()});await assert.rejects(w.client('this-tab').setup.replace({...confirmations,[key]:false}),/setup-confirmations-required/);assert.deepEqual(w.requests,[])}
});

test('the panel replaces a ledger whose stopped sessions block recovery, and the reloaded page starts on the new one',async t=>{
 const w=await world({seed:stoppedLedger()}),ui=stubFoundry(t,press('replace',confirmations)),config={actorUUIDs:[H,P],budgetSeconds:600,autoRun:false};
 w.flags.explorationSession='M';
 const page=w.runtime();await page.bind({});await page.register({socket:null});
 await assert.rejects(page.api.start(config),/unresolved-evidence-no-replay/);
 const before=w.bytes(OLD),panel=await page.api.open([H,P]),render=async app=>app._renderHTML(await app._prepareContext());
 assert.match(moreActions(await render(panel)),/<button type="button" data-recovery="replaceLedger">换新账本<\/button><span>当前账本 1 页<\/span>/);
 await panel.act('replaceLedger',null);
 assert.equal(ui.dialogs.length,1);assert.match(ui.dialogs[0].content,/旧账本里还有 5 场没结束的会话、1 个未确认的时钟。/);assert.deepEqual(ui.infos,['账本已更换，请重载页面后再准备恢复。']);
 assert.equal(w.setting(),NEW);assert.equal(w.bytes(OLD),before);assert.equal(Object.hasOwn(w.flags,'explorationSession'),false);
 // Until the page reloads it can use neither ledger, and says so instead of offering another selection.
 assert.deepEqual(await page.api.storage.status(),{state:'blocked',reason:'revision-root-changed'});
 const waiting=await render(panel);assert.match(waiting,/探索账本已切换，请重载页面后再继续。/);assert.deepEqual(actionsIn(waiting),['refresh']);
 await assert.rejects(page.api.start(config),/exploration-ledger-setup-required/);
 // Its selection went with the old ledger, so an encounter or a rest before the reload finds nothing to stop and reports nothing.
 // Clearing the flag already ran this page's updateUser hook; a player connecting runs the same one. Neither reaches for a ledger.
 page.emitHook('updateCombat',{started:true,previous:{round:0},round:1});page.emitHook('pf2e.restForTheNight',w.game.actors.get('H'));page.emitHook('userConnected',w.game.users.get('G2'));
 await new Promise(resolve=>setImmediate(resolve));assert.deepEqual(page.errors,[]);
 // The reload. This GM's flag may still name a session of the old ledger; the new one holds no such session.
 w.flags.explorationSession='M';
 const reloaded=w.runtime();await reloaded.bind({});
 const status=await reloaded.api.storage.status();assert.equal(status.state,'ready');assert.equal(status.rootUUID,NEW);assert.equal(status.revision,0);
 const reopened=await reloaded.api.open([H,P]),html=await render(reopened);
 assert.deepEqual(mainActions(html),['start']);assert.match(html,/当前账本 1 页/);assert.equal((await reloaded.api.snapshot()).session,null);
 const session=await reloaded.api.start(config);assert.equal(session.status,'running');assert.deepEqual(session.actorUUIDs,[H,P]);assert.equal(session.protocol.rootUUID,NEW);
 assert.equal(w.flags.explorationSession,session.id);assert.match(await render(reopened),/当前账本 2 页/);assert.equal(w.bytes(OLD),before);assert.deepEqual(reloaded.errors,[]);
});

test('a session flag that cannot be cleared is reported without failing the replacement',async t=>{
 const w=await world({seed:stoppedLedger()}),ui=stubFoundry(t,press('replace',confirmations));
 w.flags.explorationSession='M';w.game.user.unsetFlag=async()=>{throw Error('flag-write-failed')};
 const page=w.runtime();await page.bind({});const panel=await page.api.open([H,P]);await panel.act('replaceLedger',null);
 assert.equal(w.setting(),NEW);assert.deepEqual(ui.infos,['账本已更换，请重载页面后再准备恢复。']);assert.deepEqual(page.errors.map(error=>error.message),['flag-write-failed']);
 const reloaded=w.runtime();await reloaded.bind({});const reopened=await reloaded.api.open([H,P]);
 assert.equal(w.flags.explorationSession,'M');assert.deepEqual(mainActions(await reopened._renderHTML(await reopened._prepareContext())),['start']);
});

test('a selection left over from the replaced ledger renders as no session, and the same members can start',async t=>{
 const w=await world({seed:stoppedLedger()});stubFoundry(t);await w.client('this-tab').setup.replace(confirmations);
 const {ledger,setup}=w.client('reloaded-tab'),members=[H,P].map((actorUUID,i)=>({actorUUID,name:actorUUID,medicine:{rank:i?0:1},slugs:[],items:[],assuranceSkills:[],pool:{poolUUID:actorUUID,ready:true},hp:{value:i?1:20,max:20},focus:{value:0,max:0},wardCapacity:1,modeOfBeing:'living'}));
 const capabilities={snapshot:async uuids=>structuredClone(members.filter(member=>uuids.includes(member.actorUUID)))};
 const coordinator=createCoordinator({game:w.game,ledger,clock:{stop(){}},capabilities,providers:[],policy:chooseNext,isAuthority:()=>true,now:()=>w.game.time.worldTime,pauseGame:async()=>{},ownerOperations:{}});
 let selected='M';const app=createRecoveryPanel({game:w.game,coordinator,capabilities,storage:{status:setup.status},start:config=>coordinator.start({...config,autoRun:false}),getSessionId:()=>selected}).open([H,P]);
 assert.equal((await coordinator.snapshot('M')).session,null);
 const html=await app._renderHTML(await app._prepareContext());
 assert.deepEqual(mainActions(html),['start']);assert.match(html,/选择成员和策略，计算恢复并推进时间。/);assert.match(moreActions(html),/data-recovery="replaceLedger">换新账本/);
 const session=await coordinator.start({actorUUIDs:[H,P],autoRun:false});assert.equal(session.status,'running');selected=session.id;
 assert.deepEqual(mainActions(await app._renderHTML(await app._prepareContext())),['stop']);
});

function panelFixture(t,{ledgerStatus={state:'ready',revision:170},sessionStatus,storage=true,answer,preview=async()=>({unfinishedSessions:5,unconfirmedClocks:1}),replace=async()=>({requiresReload:true})}={}){
 const ui=stubFoundry(t,answer),calls=[],errors=[],renders=[];
 const actors=[{actorUUID:H,name:'Healer',hp:{value:20,max:20},focus:{value:0,max:0},pool:{poolUUID:H,ready:true},wardCapacity:1}];
 const data=sessionStatus?{session:{id:'S',status:sessionStatus,manual:sessionStatus==='recording',startedAt:0,cursorAt:0,goalsByPool:[{poolUUID:H,targetHP:20}]},actors,activities:[],clocks:[]}:null;
 const ledgerStorage={status:async()=>ledgerStatus,replacePreview:async()=>{calls.push(['preview']);return preview()},replace:async options=>{calls.push(['replace',options]);return replace()}};
 const app=createRecoveryPanel({game:{user:{isGM:true,getFlag:()=>({})},messages:new Map()},coordinator:{snapshot:async()=>data},capabilities:{snapshot:async()=>actors},storage:storage?ledgerStorage:undefined,getSessionId:()=>data?'S':null,onError:error=>errors.push(error)}).open([H]);
 app.render=options=>{renders.push(options);return app};
 return {app,ui,calls,errors,renders,html:async()=>app._renderHTML(await app._prepareContext())};
}

test('a ready ledger offers replacement beside its page count, without counting what would be left behind',async t=>{
 for(const sessionStatus of [undefined,'paused','complete','closed']){
  const f=panelFixture(t,{sessionStatus}),html=await f.html();
  assert.match(moreActions(html),/<button type="button" data-recovery="replaceLedger">换新账本<\/button><span>当前账本 171 页<\/span>/,String(sessionStatus));assert.deepEqual(f.calls,[]);
 }
});

for(const sessionStatus of ['running','recording'])test(`replacement is disabled while the selected session is ${sessionStatus}`,async t=>{
 assert.match(moreActions(await panelFixture(t,{sessionStatus}).html()),/<button type="button" data-recovery="replaceLedger" disabled>换新账本<\/button>/);
});

test('replacement is offered only for a ready ledger this panel can reach',async t=>{
 for(const ledgerStatus of [{state:'unconfigured'},{state:'legacy',revision:null},{state:'migration-required',revision:3},{state:'blocked',reason:'root-not-found'},{state:'blocked',reason:'revision-root-changed'}]){
  const html=await panelFixture(t,{ledgerStatus}).html();assert.ok(!actionsIn(html).includes('replaceLedger'),ledgerStatus.state);assert.doesNotMatch(html,/当前账本/);
 }
 const detached=await panelFixture(t,{storage:false}).html();assert.ok(!actionsIn(detached).includes('replaceLedger'));assert.doesNotMatch(detached,/当前账本/);
});

test('the replacement dialog states what is left behind, and replaces only with the three confirmations it collected',async t=>{
 const f=panelFixture(t,{answer:press('replace',confirmations)});await f.app.act('replaceLedger',null);
 assert.equal(f.ui.dialogs.length,1);const {content}=f.ui.dialogs[0];
 for(const line of ['旧账本里还有 5 场没结束的会话、1 个未确认的时钟。','换新之后它们不再阻挡，也不会被重放；以角色卡上现在的 HP 和世界时间为准。','旧账本原样保留为归档。新账本的名字带今天的日期和时间，清理归档时别删错。','有限药品的跨会话用量会从新账本重新算起。'])assert.ok(content.includes(`<p>${line}</p>`),line);
 for(const [name,label] of [['issuersStopped','所有旧恢复操作均已停止'],['clientsReloaded','相关客户端均已重载'],['recoveryDisabled','当前没有自动恢复正在执行']])assert.ok(content.includes(`<input type="checkbox" name="${name}" required> ${label}</label>`),name);
 assert.deepEqual(f.calls,[['preview'],['replace',confirmations]]);assert.deepEqual(f.ui.infos,['账本已更换，请重载页面后再准备恢复。']);assert.deepEqual(f.renders,[true]);
});

test('an unticked confirmation reaches storage as false, and a cancelled dialog reaches nothing',async t=>{
 const partial=panelFixture(t,{answer:press('replace',{...confirmations,clientsReloaded:false})});await partial.app.act('replaceLedger',null);
 assert.deepEqual(partial.calls[1],['replace',{...confirmations,clientsReloaded:false}]);
 for(const answer of [press('cancel'),()=>null]){const f=panelFixture(t,{answer});await f.app.act('replaceLedger',null);assert.deepEqual(f.calls,[['preview']]);assert.deepEqual(f.ui.infos,[]);assert.deepEqual(f.renders,[])}
});

test('a refused replacement opens no dialog, a failed one announces nothing, and a click reports the refusal in Chinese',async t=>{
 const active=panelFixture(t,{preview:async()=>{throw Error('replace-ledger-session-active')}});
 await assert.rejects(active.app.act('replaceLedger',null),/replace-ledger-session-active/);assert.deepEqual(active.ui.dialogs,[]);assert.deepEqual(active.calls,[['preview']]);
 const failed=panelFixture(t,{answer:press('replace',confirmations),replace:async()=>{throw Error('revision-acknowledgement-unknown')}});
 await assert.rejects(failed.app.act('replaceLedger',null),/revision-acknowledgement-unknown/);assert.deepEqual(failed.ui.infos,[]);assert.deepEqual(failed.renders,[]);
 const detached=panelFixture(t,{storage:false});await assert.rejects(detached.app.act('replaceLedger',null),/exploration-storage-unavailable/);
 const listeners=new Map();active.app._replaceHTML('',{addEventListener:(event,handler)=>listeners.set(event,handler)});
 listeners.get('click')({target:{closest:selector=>selector==='[data-recovery]'?{dataset:{recovery:'replaceLedger'}}:null}});await new Promise(resolve=>setImmediate(resolve));
 assert.equal(active.errors.length,1);assert.equal(active.errors[0].message,'仍有恢复会话在进行，请先停止再换新账本');assert.equal(active.errors[0].cause.message,'replace-ledger-session-active');
});
