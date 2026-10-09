import {createSalubriousMessagePrivacy,loadSalubriousWorkbench} from './salubrious-message-privacy.mjs';
import {createPilgrimRewards} from './sog-pilgrim-rewards.mjs';
import {createExplorationRuntime} from './exploration/runtime.mjs';
import {resolveProviderAction} from './runtime.mjs';
import {renderSalubriousCard,filterSalubriousDamageContext} from './salubrious-kiss-chat.mjs';
import {MODULE_ID,findFeature} from './rules.mjs';
import {installDailiesCompatibility} from './dailies-compat.mjs';
import {registerThirdPartyDailies} from './dailies-integration.mjs';
import {registerCampaignDailies} from './daily-feats.mjs';
import {registerUsageEvents,defaultUsageAction} from './usage-events.mjs';
import {createNativeChooser,showNativeChoice,validateNativeChoices,runDamagePipeline} from './native-context.mjs';
import {createCompanionAutomation} from './companion-automation.mjs';
import {createDualStrikeAutomation} from './dual-strike-automation.mjs';
import {createCampaignFeats} from './campaign-feats.mjs';
import {createKnowledgeAutomation,buildKnowledgePatreonRepairs,KNOWLEDGE_SOURCES} from './knowledge-automation.mjs';
import {interceptKnowledgeProbe} from './knowledge-probes.mjs';
import {createSocialAutomation} from './social-automation.mjs';
import {createThrallAutomation} from './thrall-automation.mjs';
import {createReactionChecks} from './reaction-checks.mjs';
import {registerReactionShieldWallEmptyCompatibility} from './reaction-shield-wall-empty-compat.mjs';
import {registerRoaringReactionCompatibility} from './roaring-reaction-compat.mjs';
import {createFearAutomation} from './fear-automation.mjs';
import {registerPatreonInitiativeCompatibility as installPatreonInitiativeCompatibility} from './patreon-initiative-compat.mjs';
import {createScareToDeath} from './scare-to-death.mjs';
import {createElementalMedicine} from './elemental-medicine.mjs';
import {createSalubriousCheckScope} from './salubrious-kiss-check-scope.mjs';
import {createSalubriousExecutor} from './salubrious-kiss-executor.mjs';
import {createSalubriousKiss} from './salubrious-kiss.mjs';
import {createSalubriousDamageGuard} from './salubrious-kiss-damage-guard.mjs';
import {installPatreonTreatmentCompatibility} from './patreon-treatment-compat.mjs';
import {createTranscendentDeflection} from './transcendent-deflection.mjs';
import {createDeflectionRepair} from './transcendent-deflection-repair.mjs';
import {createReactionBudget} from './reaction-budget.mjs';
import {createGlimpseCompat,glimpseWorld} from './glimpse-compat.mjs';
import {createGlimpseProvider} from './glimpse-of-redemption.mjs';
import {createSpiritualScarProvider} from './spiritual-scar.mjs';
import {createSpiritualScarFollowup} from './spiritual-scar-followup.mjs';
import {glimpseReactionSetting,canSuppressGlimpseReminder,GLIMPSE_REACTION_REASON} from './glimpse-reaction-setting.mjs';
import {registerGlimpseConfigurationEvents} from './glimpse-configuration-events.mjs';
import {createGlimpseReactionCache} from './glimpse-reaction-cache.mjs';
import {getNativeCastEvents} from './amp-cast-events.mjs';
import {createDisruptPrey} from './disrupt-prey.mjs';
import {createDisruptPreyEvents} from './disrupt-prey-events.mjs';
import {createDisruptPreyExecutor} from './disrupt-prey-executor.mjs';
import {createDisruptPreyDamageGuard} from './disrupt-prey-damage-guard.mjs';
import {createDisruptPreySourceCost} from './disrupt-prey-source-cost.mjs';
import {createDestructiveBlock} from './destructive-block.mjs';
import {createDisarmingBlock} from './disarming-block.mjs';
import {createShieldBlockEvents} from './shield-block-events.mjs';
import {resolveShieldBlockSource,validateShieldBlockSource} from './shield-block-source.mjs';
import {createDisarmContext} from './disarm-context.mjs';
import {createDisarmRegrip} from './disarm-regrip.mjs';
import {createShieldDamageAdapter} from './shield-damage-adapter.mjs';
import {notifyNativeIWRStatus} from './native-iwr-status.mjs';
import {verifyNativeIWRBridge} from './native-iwr-verification.mjs';
import {verifyManualPoolProviders} from './exploration/manual-pool-provider.mjs';
import {createRuneTransfer,registerRuneTransferRuleElement} from './rune-transfer.mjs';
import {preserveDamageBypassOnAlter} from './native-damage-components.mjs';
import {createMetapowerProvider,preserveMetapowerOnAlter} from './metapower/provider.mjs';
import {createEldamonElectricityProvider,preserveElectricityOnAlter} from './eldamon-electricity-provider.mjs';
import {createEldamonBasicSettlement} from './eldamon-basic-settlement.mjs';
import {createFortressRuleCompatibility,buildFortressPatreonRepairs,fortressRuleCompatibilityEnabled} from './fortress-rule-compat.mjs';
import {createEldamonDataRepair} from './eldamon-data-repair.mjs';
import {createMedicActions} from './medic-actions.mjs';
import {createBardFamiliarProvider} from './bard-familiar.mjs';
import {createDesperatePrayerProvider} from './desperate-prayer.mjs';
import {createHalflingLuckProvider} from './halfling-luck.mjs';
import {createForceBarrageBridge} from './force-barrage.mjs';
import {createRoaringEffects} from './roaring-effects.mjs';
import {createRoaringApplause} from './roaring-applause.mjs';
import {createRoaringSustain} from './roaring-sustain.mjs';
import {createDefensiveAdvance} from './defensive-advance.mjs';
import {buildDefensiveAdvancePatreonRepairs,defensiveAdvanceStartupCompatibility} from './defensive-advance-compat.mjs';
import {createEldamonVoltageProvider} from './eldamon-voltage-executor.mjs';
import {createSpellCombination} from './spell-combination.mjs';
import {createNativeTargetSaves} from './native-target-saves.mjs';
import {createAvAutomation,buildAvPatreonRepairs} from './av-automation.mjs';
import {createPartyAutomation,buildPartyPatreonRepairs,PARTY_SOURCES} from './party-automation.mjs';
import {createConfigurationMaintenance} from './config-maintenance.mjs';
import {createMaintenance} from './maintenance.mjs';
import {createDirtyMaintenance,isUnrelatedMaintenanceUpdate,COSMETIC_UPDATE_FIELDS} from './maintenance-events.mjs';
import {createActivityResultLifecycle} from './activity-result-lifecycle.mjs';
import {createWeaponSurgeAutomation} from './weapon-surge.mjs';
import {createNativeOwnerOperations} from './native-owner-operations.mjs';
import {createCycleCoordinator} from './cycle-coordinator.mjs';
import {createCycleAutomation,addCycleReactionButtons} from './cycle-automation.mjs';
import {createPanel,createUsageExecutor,executeActorAction,onFullRest,expireCycles,requireOwner,SerialActions} from './runtime.mjs';
let socket,coordinator,cycle,maintenance,providers=[];
const report=e=>{console.error(MODULE_ID,e);ui.notifications.error(e.message);};
const glimpseCompat=createGlimpseCompat({game,fromUuid});
glimpseCompat.register({Hooks});
Hooks.once('init',()=>game.settings.register(MODULE_ID,'configurationBackups',{scope:'world',config:false,type:Array,default:[]}));
Hooks.once('init',()=>game.settings.register(MODULE_ID,'explorationLedgerUUID',{scope:'world',config:false,type:String,default:''}));
Hooks.once('setup',()=>{installDailiesCompatibility(game);registerThirdPartyDailies(game);registerCampaignDailies(game);registerRuneTransferRuleElement(game);});
Hooks.once('socketlib.ready',()=>{
 socket=socketlib.registerModule(MODULE_ID);
 socket.register('native-choice',async function(payload){
  if(this.socketdata.userId!==game.users.activeGM?.id)throw Error('只有当前主GM可以请求规则选择。');
  const actor=await fromUuid(payload.actorUuid);requireOwner(actor,game.user);
  return showNativeChoice({title:payload.title,choices:validateNativeChoices(payload.choices)});
 });
 socket.register('action',async function(actorUuid,action,payload){
  try{
   if(game.user!==game.users.activeGM)throw Error('操作必须由当前主GM执行。');
   const actor=await fromUuid(actorUuid),user=game.users.get(this.socketdata.userId);
   return {ok:true,value:await executeActorAction(actor,action,payload,user)};
  }catch(error){return {ok:false,error:error.message};}
 });
 for(const method of ['claim','complete'])socket.register(`cycle-${method}`,async function(payload){
  try{
   if(game.user!==game.users.activeGM)throw Error('操作必须由当前主GM执行。');
   return {ok:true,value:await coordinator[method](payload,game.users.get(this.socketdata.userId))};
  }catch(error){return {ok:false,error:error.message};}
 });
 socket.register('cycle-trait',async function(actorUuid,choices){
  if(!game.users.get(this.socketdata.userId)?.isGM)throw Error('只有执行GM可以请求反应特征选择。');
  const actor=await fromUuid(actorUuid);requireOwner(actor,game.user);
  return foundry.applications.api.DialogV2.wait({window:{title:`循环能量 · ${actor.name}`},content:'<p>触发效果同时具有命能和虚能，请选择循环的能量。</p>',buttons:choices.filter(c=>['void','vitality'].includes(c.trait)).map(c=>({action:c.trait,label:c.trait==='void'?'虚能':'命能',callback:()=>c.trait})),rejectClose:false});
 });
});
const rpc=async(method,...args)=>{
 if(!socket||!game.users.activeGM)throw Error('需要在线GM与socketlib才能自动结算。');
 const result=await socket.executeAsUser(method,game.users.activeGM.id,...args);
 if(!result.ok)throw Error(result.error);return result.value;
};
const request=async(actorUuid,action,payload)=>{
 return rpc('action',actorUuid,action,payload);
};
const panel=createPanel(request);
const open=actor=>{if(!game.user.isGM)throw Error('维护面板仅供GM排错；请直接使用技能或每日准备。');return panel(actor);};
async function useCycleFromCard(actor,source){
 requireOwner(actor,game.user);
 const item=findFeature(actor,'cycle');if(!item)throw Error('没有循环能量。');
 const message=await item.toMessage(null,{create:false,actualUse:true});
 message.updateSource({[`flags.${MODULE_ID}.cycleDamage`]:source});
 return ChatMessage.create(message.toObject());
}
Hooks.once('ready',async()=>{
 const choose=createNativeChooser({game,send:(userId,payload)=>socket.executeAsUser('native-choice',userId,payload)});
 const ownerRolls=Object.fromEntries(['campaign','av','companion','knowledge','social','reaction','rest'].map(scope=>[scope,createNativeOwnerOperations({game,fromUuid,scope})]));
 const targetSaves=Object.fromEntries(['spell-combination','spiritual-scar'].map(scope=>[scope,createNativeTargetSaves({game,fromUuid,choose,scope})]));
 const activityResults=createActivityResultLifecycle({game,getRollContext:roll=>cycle?.getRollContext(roll),onError:report});
 const weaponSurge=createWeaponSurgeAutomation({game});
 const advanceStartup=defensiveAdvanceStartupCompatibility({game,rules:game.modules.get('patreon-v3')?.active?game.settings.get('patreon-v3','rulesV3'):null});
 const defensiveAdvance=createDefensiveAdvance({game,fromUuid,choose,startupCompatibility:advanceStartup,onError:report});
 let glimpse,roaring,roaringReactionCompatibility;
 const reactionRestriction=game.world?.id==='ujx5r8oipw7ercdr'&&game.system?.id==='pf2e'?actor=>roaring?.reactionRestriction(actor)??{status:'manual',sources:[{sourceNonce:null,status:'manual',reason:'provider-initializing'}]}:undefined;
 const reactionBudget=createReactionBudget({game,fromUuid,reactionRestriction,onError:report,handlesGlimpse:actor=>glimpse?.handlesActor(actor)??false});
 await glimpseCompat.initialize({game}).catch(report);
 glimpse=createGlimpseProvider({game,fromUuid,reactionRestriction,compat:glimpseCompat,getRollContext:roll=>cycle?.getRollContext(roll),onError:report});
 const nativeCasts=getNativeCastEvents({game,fromUuid});
 const disruptDamage=createDisruptPreyDamageGuard({game,getRollContext:roll=>cycle?.getRollContext(roll)});
 const disruptExecutor=createDisruptPreyExecutor({game,fromUuid,Hooks,authorizeDamage:context=>disruptDamage.authorize(context)});
 const sourceCost=createDisruptPreySourceCost({game});let disrupt;
 const disruptEvents=createDisruptPreyEvents({game,fromUuid,handleConfirmed:event=>disrupt.handleConfirmed(event),onSourceStopped:(context,result)=>sourceCost.stopped(context,result),onError:report});
 disrupt=createDisruptPrey({game,fromUuid,choose,reactionRestriction,validateSource:(event,context)=>disruptEvents.validateSource(event,context),performStrike:context=>disruptExecutor.performStrike(context),onError:report});
 const destructiveBlock=createDestructiveBlock({game,fromUuid,choose});
 let shieldEvents;
 const disarmingBlock=createDisarmingBlock({game,fromUuid,choose,validateConfirmed:event=>shieldEvents.validateConfirmed(event),onError:report});
 shieldEvents=createShieldBlockEvents({game,fromUuid,resolveSource:resolveShieldBlockSource,validateSource:validateShieldBlockSource,onConfirmed:event=>disarmingBlock.handleConfirmed(event),onError:report});
 const disarmContext=createDisarmContext({game,processor:disarmingBlock});
 const disarmRegrip=createDisarmRegrip({game,fromUuid,processor:disarmingBlock,chooseOwner:choose,onError:report});
 const salubriousMessagePrivacy=createSalubriousMessagePrivacy({game,Hooks});
 const workbenchPrivacy=await loadSalubriousWorkbench({game});
 if(workbenchPrivacy.ready)salubriousMessagePrivacy.enableWorkbench(workbenchPrivacy);
 let nativeBridgeVerification=Object.freeze({ready:false,reason:'system-source-unavailable'}),pf2eSource;
 try{const response=await fetch('systems/pf2e/pf2e.mjs',{cache:'no-store',signal:AbortSignal.timeout(10000)});if(response.ok){pf2eSource=new Uint8Array(await response.arrayBuffer());nativeBridgeVerification=await verifyNativeIWRBridge({game,source:pf2eSource});}}catch{/* Report via the feature diagnostic; other providers still initialize. */}
 if(pf2eSource&&game.modules.get('pf2e-toolbelt')?.active){
  try{const response=await fetch('modules/pf2e-toolbelt/scripts/main.js',{cache:'no-store',signal:AbortSignal.timeout(10000)});if(response.ok)await verifyManualPoolProviders({game,pf2eSource,toolbeltSource:new Uint8Array(await response.arrayBuffer())});}catch{/* Missing qualification keeps only the shared-pool adapter unavailable. */}
 }
 const shieldAdapter=createShieldDamageAdapter({game,nativeBridgeVerification,onError:report,createMessageMiddleware:salubriousMessagePrivacy.createMessageMiddleware});
 const scar=game.world?.id==='ujx5r8oipw7ercdr'&&game.system?.id==='pf2e'?createSpiritualScarProvider({game,fromUuid,nativeAdapter:shieldAdapter,reactionRestriction,getRollContext:roll=>cycle?.getRollContext(roll),followup:createSpiritualScarFollowup({game,fromUuid,Hooks,runTargetSave:targetSaves['spiritual-scar'].run,onError:report}),onError:report}):null;
 const deflection=createTranscendentDeflection({game,fromUuid,choose,reactionRestriction,getRollContext:roll=>cycle?.getRollContext(roll),nativeBridgeAvailable:shieldAdapter.nativeBridgeAvailable,onError:report});
 shieldAdapter.addNativeInterceptor(deflection.interceptNative,{matches:deflection.hasNativePlan});
 const deflectionRepair=createDeflectionRepair({game,fromUuid,onError:report});
 const elementalMedicine=createElementalMedicine({game,fromUuid,onError:report});
 const exploration=createExplorationRuntime({game,Hooks,fromUuid,nativeCasts,onError:report});
 const salubriousCheckScope=createSalubriousCheckScope({game,isExplorationContext:exploration.ownerOperations.isActivityContext});
 const salubriousDamage=createSalubriousDamageGuard({game,messagePrivacy:salubriousMessagePrivacy,getRollContext:roll=>cycle?.getRollContext(roll),isExplorationContext:exploration.ownerOperations.isActivityContext});
 const salubriousExecutor=createSalubriousExecutor({game,fromUuid,Hooks,checkScope:salubriousCheckScope,authorizeDamage:salubriousDamage.authorize,hpPools:exploration.hpPools,isExplorationContext:exploration.ownerOperations.isActivityContext});
 const salubriousKiss=createSalubriousKiss({game,fromUuid,choose,executor:salubriousExecutor,validateRefocusNote:salubriousMessagePrivacy.validateRefocusNote,isExplorationContext:exploration.ownerOperations.isActivityContext});
 await exploration.bind({checkScope:salubriousCheckScope,damageGuard:salubriousDamage,salubriousKiss});
 let treatmentDiagnostic=Object.freeze({ready:false,installed:false,reason:'initializing',dependency:null});
 const treatmentRefocus={matchesActor:actor=>treatmentDiagnostic.ready&&salubriousKiss.matchesActor(actor),
  onRefocus:async event=>{if(!treatmentDiagnostic.ready)throw Error('仙露三吻的原生检定兼容尚未就绪。');return salubriousKiss.onRefocus(event)}};
 const runeTransfer=createRuneTransfer({game,fromUuid,choose,onError:report});
 const campaign=createCampaignFeats({game,fromUuid,choose,runNative:ownerRolls.campaign.run,onError:report});
 const fear=createFearAutomation({game,fromUuid,choose,reactionRestriction,onError:report});
 const familiar=createBardFamiliarProvider({game,fromUuid,onError:report});
 const prayer=game.world?.id==='ujx5r8oipw7ercdr'?createDesperatePrayerProvider({game,fromUuid,choose,onError:report,castEvents:nativeCasts}):null;
 const halflingLuck=game.world?.id==='ujx5r8oipw7ercdr'&&game.system?.id==='pf2e'?createHalflingLuckProvider({game,fromUuid,choose:showNativeChoice,onError:report}):null;
 const forceBarrage=game.world?.id==='ujx5r8oipw7ercdr'&&game.system?.id==='pf2e'?createForceBarrageBridge({game,fromUuid,nativeCasts,onError:report}):null;
 roaring=forceBarrage?createRoaringApplause({game,fromUuid,nativeCasts,effects:createRoaringEffects({game,fromUuid,onError:report}),onError:report,onManual:()=>ui.notifications.warn('本次轰然喝彩需要GM核对结果或时长，请查看原施法卡。')}):null;
 const roaringSustain=roaring?createRoaringSustain({game,fromUuid,provider:roaring,reactionCompatibility:()=>({owned:true,checker:roaringReactionCompatibility?.ready()===true}),onError:report}):null;
 if(forceBarrage)nativeCasts.addCastMiddleware(forceBarrage.interceptCast);
 if(roaring)nativeCasts.addCastMiddleware(roaring.interceptCast);
 if(prayer){nativeCasts.addActorMatcher(prayer.isManagedActor);nativeCasts.addConsumePolicy(prayer.consumePolicy);nativeCasts.addCastMiddleware(prayer.interceptCast);}
 const prayerCheck=(native,...args)=>interceptKnowledgeProbe((...checked)=>prayer?prayer.interceptCheck(native,...checked):native(...checked),...args);
 let metapower,electricity;
 const voltage=createEldamonVoltageProvider({game,fromUuid,onError:report,getRollContext:roll=>cycle?.getRollContext(roll),observe:(...args)=>metapower.observe(...args),onRefresh:context=>electricity.onRefresh(context)});
 electricity=createEldamonElectricityProvider({game,fromUuid,reactionRestriction,onError:report,refreshOutsideEncounter:voltage.refreshOutsideEncounter,useOwnedAction:(item,event)=>metapower.observe({actor:item.actor,item},()=>game.toolbelt.api.actionable.useAction(event,item))});
 metapower=createMetapowerProvider({game,fromUuid,onError:report,supportsOriginalUse:item=>providers.some(p=>p.resolveAction?.(item)?.startsWith('medic:')||['glimpse:use','defensive-advance'].includes(p.resolveAction?.(item))),beforeChannel:electricity.beforeChannel,validateSelection:electricity.validateSelection,interceptDamageMessage:electricity.interceptDamageMessage,onCommittedChannel:async context=>{await voltage.onCommittedChannel(context);await electricity.onCommittedChannel(context)}});
 nativeCasts.addCastMiddleware(({item,options},native)=>options.consume===false||options.message===false?native():metapower.observe({actor:item.actor,entry:'spell'},native));
 providers=[createCompanionAutomation({game,fromUuid,choose,runNative:ownerRolls.companion.run,onError:report,wrapStrike:(strike,actor)=>providers.reduce((s,p)=>p.wrapStrike?.(s,actor)??s,strike)}),createDualStrikeAutomation({game,fromUuid,choose,onError:report}),runeTransfer,campaign,createKnowledgeAutomation({game,fromUuid,choose,runNative:ownerRolls.knowledge.run,onError:report}),createAvAutomation({game,fromUuid,choose,runNative:ownerRolls.av.run,onError:report,refocusSubscribers:[treatmentRefocus,exploration.refocusSubscriber],refocusPrivacy:salubriousMessagePrivacy,explorationRefocus:exploration.refocusAdapter}),createPartyAutomation({game,fromUuid,choose,onError:report}),createSocialAutomation({game,fromUuid,choose,runNative:ownerRolls.social.run,onError:report}),createThrallAutomation({game,fromUuid,choose,onError:report}),createReactionChecks({game,fromUuid,choose,runNative:ownerRolls.reaction.run,nativeInvocation:ownerRolls.reaction.getInvocation,reactionRestriction,onError:report,halflingLuck,nativeCheckMiddleware:(wrapped,...args)=>prayerCheck((...checked)=>familiar.interceptCheck((...accompanied)=>electricity.interceptCheck((...electric)=>metapower.interceptCheck((...next)=>weaponSurge.interceptCheck((...surged)=>salubriousCheckScope.interceptCheck(wrapped,...surged),...next),...electric),...accompanied),...checked),...args)}),fear,createScareToDeath({game,fromUuid,choose,onError:report}),createSpellCombination({game,fromUuid,choose,runTargetSave:targetSaves['spell-combination'].run,onError:report,afterAttack:message=>campaign.processCheck(message)}),deflection,destructiveBlock,disarmingBlock,disarmRegrip,shieldEvents,salubriousKiss];
 const onRulesChanged=()=>{if(fortressRuleCompatibilityEnabled(game)&&game.modules.get('patreon-v3')?.active)ui.notifications.warn('Patreon 规则已修复；请所有已在线客户端刷新一次，使规则修复生效。',{permanent:true})};
 const configuration=createConfigurationMaintenance({game,onRulesChanged,repairs:[buildAvPatreonRepairs,buildPartyPatreonRepairs,buildKnowledgePatreonRepairs,rules=>buildFortressPatreonRepairs(rules,{game}),...game.world?.id==='ujx5r8oipw7ercdr'?[buildDefensiveAdvancePatreonRepairs]:[]],settings:[{module:'pf2e-ranged-combat',key:'postActionToChat',value:2,when:g=>Array.from(g.actors.party?.members??[]).some(a=>[KNOWLEDGE_SOURCES.monster,KNOWLEDGE_SOURCES.hunt].every(source=>a.items.some(i=>i.sourceId===source))),reason:'猎物指定保留完整原生技能卡，供怪物猎手知识联动读取原始操作者与目标。'},{module:'pf2e-reaction',key:'builtinReactionsEnabled',when:g=>['-','sog','pnvfcgjbf2cjp7gz','ujx5r8oipw7ercdr','team-automation-qa2'].includes(g.world?.id),transform:value=>Array.isArray(value)?value.filter(slug=>slug!=='disarming-block'):value,reason:'卸武格挡改由实际格挡回执接原生自由动作缴械，避免重复提示或再次收取反应。'}]});
 await configuration().catch(report);
 if(glimpseWorld(game)){
  const cache=createGlimpseReactionCache({game});await cache.initialize().catch(report);
  const actors=()=>[...game.actors.contents,...game.scenes.contents.flatMap(scene=>scene.tokens.contents.map(token=>token.actor).filter(Boolean))];
  const reconcile=createConfigurationMaintenance({game,settings:[{module:'pf2e-reaction',key:'builtinReactionsEnabled',when:glimpseWorld,transform:(value,g)=>glimpseReactionSetting(value,g,cache.ready()&&canSuppressGlimpseReminder(actors(),glimpse)),reason:GLIMPSE_REACTION_REASON}]});
  await registerGlimpseConfigurationEvents({game,Hooks,reconcile,onError:report}).reconcileNow();
 }
 providers.unshift(activityResults,weaponSurge,glimpse,...scar?[scar]:[],voltage,electricity);
 providers.push(metapower,createEldamonDataRepair({game}),createMedicActions({game,fromUuid,choose,onError:report}),familiar,defensiveAdvance,...prayer?[prayer]:[]);
 const electricityBasic=createEldamonBasicSettlement({game,fromUuid,basicUse:electricity.basicUse,apply:payload=>electricity.confirmedAction(payload),onError:report});
 providers.push(createFortressRuleCompatibility({game}),electricityBasic);
 if(halflingLuck)providers.push(halflingLuck);
 if(forceBarrage)providers.push(forceBarrage);
 // Prayer subscribes first so a genuine Sustain still closes its action window.
 if(roaring)providers.push(roaring,roaringSustain);
 coordinator=createCycleCoordinator({game,
  chooseTrait:(actor,user,choices)=>socket.executeAsUser('cycle-trait',user.id,actor.uuid,choices),
  onEffect:(actor,claim,user)=>executeActorAction(actor,'cycle',{damageType:claim.damageType,triggerConfirmed:true},user,{cycleTiming:claim.timing}),
 });
 cycle=createCycleAutomation({
  onClaim:context=>rpc('cycle-claim',context),
  onComplete:result=>{
   const actor=fromUuidSync(result.context.actorUuid);
   if(actor?.type==='character'&&findFeature(actor,'cycle'))return rpc('cycle-complete',result);
  },onError:report,
 });
 libWrapper.register(MODULE_ID,'CONFIG.Actor.documentClass.prototype.applyDamage',function(wrapped,params){
  const manualPoolFrame=exploration.manualPoolApplication.captureFrame(this,params);
  const applyNative=next=>exploration.manualPoolApplication.applyNativeDamage(this,next,p=>activityResults.applyNativeDamage(this,p,wrapped),manualPoolFrame);
  // Consume the exact private params grant before any normalization or spread.
  return salubriousDamage.applyDamage(this,params,(treatmentApproved,assertSalubrious)=>disruptDamage.applyDamage(this,treatmentApproved,(approved,assertNative)=>{
   const source=cycle.getRollContext(approved.damage);
   const actual=source?{...approved,rollOptions:[...new Set([...(approved.rollOptions??[]),`${MODULE_ID}:source:${source.messageId}:${source.rollIndex}`])]}:approved;
   return runDamagePipeline({actor:this,params:actual,providers,apply:p=>reactionBudget.applyDamage(this,p,next=>shieldAdapter.applyDamage(this,next,final=>shieldEvents.wrapNativeDamage(this,final,native=>cycle.applyDamage(this,finalParams=>shieldAdapter.withNativeFrame(this,finalParams,checkedParams=>{assertNative();assertSalubrious(this,checkedParams);return glimpse.wrapNativeDamage(this,checkedParams,p=>scar?scar.wrapNativeDamage(this,p,applyNative):applyNative(p))}),native)),destructiveBlock.planFor(next))),onError:report});
  }));
 },'WRAPPER');
 libWrapper.register(MODULE_ID,'CONFIG.ChatMessage.documentClass.create',function(wrapped,data,...args){return exploration.manualPoolApplication.observeCreate(wrapped,data,...args)},'WRAPPER');
 const rollIndex=CONFIG.Dice.rolls.findIndex(cls=>cls.name==='DamageRoll');
 if(rollIndex<0)throw Error('未找到PF2e DamageRoll，循环能量无法接入。');
 libWrapper.register(MODULE_ID,`CONFIG.Dice.rolls.${rollIndex}.prototype.alter`,function(wrapped,...args){return preserveElectricityOnAlter(this,preserveMetapowerOnAlter(this,preserveDamageBypassOnAlter(this,cycle.alterDamageRoll(this,wrapped,...args),{multiplier:args[0]??1,addend:args[1]??0})))},'WRAPPER');
 for(const message of game.messages)cycle.recordDamageMessage(message);
 const legacyUsage=createUsageExecutor({cycleUse:(actor,message,user)=>coordinator.use(actor,message,user),runNative:ownerRolls.rest.run}),usageQueue=new SerialActions();
 const resolveAction=item=>defaultUsageAction(item)??resolveProviderAction(providers,item);
 reactionBudget.register({Hooks,socket});
 disruptExecutor.register({socket});
 salubriousExecutor.register({socket});
 if(['-','sog','pnvfcgjbf2cjp7gz','team-automation-qa2'].includes(game.world?.id))disruptEvents.register({Hooks,libWrapper,socket,castEvents:nativeCasts});
 shieldAdapter.register({libWrapper});
 deflectionRepair.register({Hooks,socket});
 elementalMedicine.register({Hooks,socket});
 for(const owner of Object.values(ownerRolls))owner.register({Hooks,socket});
 for(const owner of Object.values(targetSaves))owner.register({Hooks,socket});
 elementalMedicine.registerDailies();
 for(const p of providers)p.register?.({Hooks,libWrapper,socket,onError:report});
 if(roaring)await roaring.reconcile().catch(report);
 disarmContext.register();
 // Strike objects are prepared before ready. Rebuild them once so wrappers also
 // cover actors that needed no persistent data repair on this login.
 for(const actor of game.actors)if(actor.type==='character')actor.reset();
 registerUsageEvents({game,Hooks,resolveAction,requiresActualUse:(item,action)=>providers.some(p=>p.requiresActualUse?.(item,action)),observeItemUse:async(item,native,{entry}={})=>{if(prayer?.resolveAction(item))prayer.beforeUse(item);else await prayer?.beforeAction(item.actor);familiar.beforeUse(item);await halflingLuck?.beforeUse(item);scar?.beforeUse(item);return entry!=='native-sheet'&&["feat","action"].includes(item.type)?metapower.observe({actor:item.actor,item},native):native()},captureUsage:(item,context)=>Object.assign({},nativeCasts.captureUsage(item,context),...providers.map(p=>p.captureUsage?.(item,context))),onMessageOutcome:(item,options,outcome)=>nativeCasts.captureMessageOutcome(item,options,outcome),tracksFrequency:item=>defaultUsageAction(item)==='breath'||item.sourceId===PARTY_SOURCES.clue||resolveAction(item)==='knowledge:devise'||providers.some(p=>p.tracksFrequency?.(item)),
  executeUsage:ctx=>usageQueue.run(ctx.actor.uuid,async()=>{if(ctx.action!=='rune-transfer:select')await runeTransfer.ensureReady(ctx.actor,ctx.user);const provider=providers.find(p=>p.resolveAction?.(ctx.item)===ctx.action);return provider?provider.executeUsage(ctx):legacyUsage(ctx)}),onError:report});
 const legacyMaintenance=createMaintenance({game,repair:actor=>executeActorAction(actor,'repair',{},game.user)}),maintainQueue=new SerialActions();
 const maintenanceEvents=createDirtyMaintenance({enabled:()=>game.user?.id===game.users.activeGM?.id,onError:report,run:async scope=>{
  if(game.user?.id!==game.users.activeGM?.id)return;
  const party=Array.from(game.actors.party?.members??[]).filter(a=>a.type==='character'&&!a.flags?.[MODULE_ID]?.autoRepairDisabled);
  const targets=scope?party.filter(a=>scope.has(a.uuid)):party;
  for(const target of targets)await maintainQueue.run(target.uuid,async()=>{await legacyMaintenance(target);for(const p of providers)await p.maintain?.(target)});
 }});
 maintenance=actor=>maintenanceEvents.request(actor?.uuid??null);
 let patreonInitiativeCompatibility;
 const registerPatreonCompatibility=async()=>patreonInitiativeCompatibility=await installPatreonInitiativeCompatibility({game,Hooks,isProviderReady:()=>providers.includes(fear)});
 await registerPatreonCompatibility();
 const reactionShieldWall=await registerReactionShieldWallEmptyCompatibility({game,Hooks});
 if(roaring)roaringReactionCompatibility=await registerRoaringReactionCompatibility({game,query:reactionRestriction});
 const reactionShieldWallDiagnostic=Object.freeze({status:reactionShieldWall.status,reason:reactionShieldWall.reason??null,scope:reactionShieldWall.scope??null,sourceSHA256:reactionShieldWall.sourceSHA256??null,callbackSHA256:reactionShieldWall.callbackSHA256??null,hook:reactionShieldWall.hook?Object.freeze({...reactionShieldWall.hook}):null});
 // All existing Check providers are registered first. This bridge changes only
 // the verified Patreon entry.fn; it does not reorder or register Check again.
 try{
  const compatibility=await installPatreonTreatmentCompatibility({game,libWrapper,scope:salubriousCheckScope});
  treatmentDiagnostic=Object.freeze({ready:workbenchPrivacy.ready&&compatibility.installed===true,workbench:workbenchPrivacy.ready?workbenchPrivacy.profile:null,installed:compatibility.installed===true,reason:workbenchPrivacy.ready?compatibility.reason??null:workbenchPrivacy.reason,dependency:compatibility.dependency?Object.freeze({...compatibility.dependency}):null});
 }catch(error){treatmentDiagnostic=Object.freeze({ready:false,installed:false,reason:String(error.message??error),dependency:null});report(error);}
 game.modules.get(MODULE_ID).api={open,request,version:game.modules.get(MODULE_ID).version,repairActiveParty:()=>maintenance(),eldamon:{getContinuations:voltage.getContinuations,continueActivity:voltage.continueActivity,useRefresh:voltage.useRefresh,continueChain:electricity.continueChain,declareShieldTrigger:electricityBasic.declareShieldTrigger,clearMetapower:async(actorOrUuid,options)=>metapower.clearArmed(typeof actorOrUuid==='string'?await fromUuid(actorOrUuid):actorOrUuid,options)},nativeDamageIWR:async(...args)=>{const handled=await shieldAdapter.nativeDamageIWR(...args);electricity.observeNativeIWR(...args,handled);return handled},get nativeIWRCompatibility(){return shieldAdapter.nativeBridgeDiagnostic()},get patreonInitiativeCompatibility(){return patreonInitiativeCompatibility},registerPatreonInitiativeCompatibility:registerPatreonCompatibility,get salubriousKiss(){return treatmentDiagnostic},get defensiveAdvance(){return defensiveAdvance.diagnostic},get glimpseOfRedemption(){return {ready:glimpse.ready()}},get roaringApplause(){return {enabled:!!roaring,...roaring?.diagnostic(),ownedReactionConsumers:!!roaring,reactionChecker:{status:roaringReactionCompatibility?.status??'unavailable',ready:roaringReactionCompatibility?.ready()===true,reason:roaringReactionCompatibility?.reason??null}}},get reactionShieldWallCompatibility(){return reactionShieldWallDiagnostic}};
 game.modules.get(MODULE_ID).api.exploration=exploration.api;
 const pilgrim=await createPilgrimRewards({game,fromUuid,choose,onError:error=>console.error(MODULE_ID,'pilgrim',error)});
 if(pilgrim.enabled){pilgrim.register({Hooks,socket});game.modules.get(MODULE_ID).api.pilgrim=pilgrim;}
 await exploration.register({socket});
 notifyNativeIWRStatus({game,diagnostic:shieldAdapter.nativeBridgeDiagnostic(),warn:message=>ui.notifications.warn(message)});
 await maintenance().catch(report);
 await elementalMedicine.maintain().catch(report);
});
Hooks.on('getHeaderControlsApplicationV2',(app,controls)=>{
 const actor=app.actor??app.document;
 if(game.user.isGM&&actor?.documentName==='Actor'&&actor.type==='character')controls.push({action:'explorationRecovery',label:'探索恢复',icon:'fa-solid fa-clock',onClick:()=>game.modules.get(MODULE_ID).api.exploration.open()});
 if(game.user.isGM&&actor?.documentName==='Actor'&&actor.type==='character')controls.push({action:'thirdPartyAutomation',label:'第三方维护',icon:'fa-solid fa-wand-magic-sparkles',onClick:()=>open(actor)});
});
Hooks.on('getActorSheetHeaderButtons',(app,buttons)=>{
 if(game.user.isGM&&app.actor?.type==='character')buttons.unshift(
  {label:'探索恢复',class:'exploration-recovery',icon:'fa-solid fa-clock',onclick:()=>game.modules.get(MODULE_ID).api.exploration.open()},
  {label:'第三方维护',class:'third-party-automation',icon:'fa-solid fa-wand-magic-sparkles',onclick:()=>open(app.actor)}
 );
});
Hooks.on('pf2e.restForTheNight',actor=>onFullRest(actor,request).catch(e=>ui.notifications.error(e.message)));
Hooks.on('updateCombat',(combat,changes={})=>{if('round'in changes||'turn'in changes)expireCycles(combat).catch(e=>ui.notifications.error(e.message));});
Hooks.on('deleteCombat',combat=>expireCycles(combat,true).catch(e=>ui.notifications.error(e.message)));
Hooks.on('createItem',item=>maintenance?.(item.actor).catch(report));
Hooks.on('updateActor',actor=>{if(actor.type==='party')maintenance?.().catch(report);});
Hooks.on('updateSetting',setting=>{if(setting.key==='pf2e.activeParty')maintenance?.().catch(report);});
Hooks.on('updateUser',(_user,changes={})=>{if(!isUnrelatedMaintenanceUpdate(changes,[...COSMETIC_UPDATE_FIELDS,'color','avatar']))maintenance?.().catch(report);});
Hooks.on('createChatMessage',message=>cycle?.recordDamageMessage(message));
Hooks.on('updateChatMessage',message=>{
 cycle?.recordDamageMessage(message);
 if(game.user===game.users.activeGM&&message.flags?.pf2e?.appliedDamage?.isReverted)coordinator?.undo(message,game.user).catch(report);
});
Hooks.on('getChatMessageContextOptions',(_app,entries)=>filterSalubriousDamageContext(game,entries));
Hooks.on('renderChatMessageHTML',(message,html)=>{
 renderSalubriousCard(message,html);
 cycle?.recordDamageMessage(message);
 if(!message.isDamageRoll)return;
 const targets=canvas.tokens?.placeables?.filter(token=>token.actor?.type==='character').map(token=>({actor:token.actor,token:token.document}))??[];
 addCycleReactionButtons(message,html,{targets,onUse:useCycleFromCard,onError:report});
});
