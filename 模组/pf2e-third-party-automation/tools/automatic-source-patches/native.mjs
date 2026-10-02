import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {NATIVE_IWR_PROFILES,NATIVE_DAMAGE_SHAPES,SHARED_MANUAL_SHAPE as shape,NATIVE_IWR_BRIDGE_PROTOCOL,isAutomaticBatchDescriptor,isAutomaticToolDescriptor} from '../../scripts/native-iwr-profiles.mjs';
import {sourceText,nativeMethod,batchRegion,flatRegion,stackingRegion,toolRegions,bridgeStatement,automaticDescriptor,observerRegion,TOOL_RECEIVE,TOOL_RECEIVE_PATCHED,TOOL_FORWARD,TOOL_FORWARD_PATCHED} from '../../scripts/native-source-shapes.mjs';

const hash=value=>createHash('sha256').update(value).digest('hex');
const components={
 'native-manual-pool-batch/observer.js':'476a7251030f10673661e89a4f76dd89a1873e6e1c2c5e72d719f787b2df9665',
 'native-manual-pool-static-receiver/observer.js':'141051a2099816bdedaf688ecfc164030fdfe8edf9ce81fd71b409bb2eff9cab',
 'toolbelt-manual-pool/observer.js':'135e1a7e855765e9a7c7d0d3e39b883c604e5536b7202286e765db92ce53fcb1'
};
const read=file=>{const bytes=fs.readFileSync(new URL('../'+file,import.meta.url));if(hash(bytes)!==components[file])throw Error('native-source-seam-component');return bytes.toString('utf8')};
const once=(s,a,b,reason='native-source-seam')=>{if(s.split(a).length!==2)throw Error(reason+'-ambiguous');return s.replace(a,b)};
const legacyPF='9be357c96dff3d0790edcb0d7889db98cfded0f41f34fd161233ea0bdd0dab11',legacyTB='6937743860c8b2fbeef1037bb99792307cb3fbe7a2540ae0b1ac7010a16a5d90';
const methodAnchor='\tasync applyDamage({',health='\t\tlet ae = this.calculateHealthDelta({';
function bridgeLine(descriptor){return '\tstatic thirdPartyNativeIWRBridge = Object.freeze('+JSON.stringify(descriptor).slice(0,-1)+',"applyDamage":this.prototype.applyDamage});'}
export function buildNativeBridge({source,version}={}){
 let text=sourceText(source),method=nativeMethod(text),digest=hash(method),contract=NATIVE_DAMAGE_SHAPES.find(p=>p.originalSHA256===digest||p.applyDamageSHA256===digest);
 if(!contract)throw Error('native-bridge-seam-method');
 const hasBridge=text.includes('thirdPartyNativeIWRBridge'),hasCallback=text.includes('thirdPartyIWR');
 if(digest===contract.originalSHA256&&(hasBridge||hasCallback))throw Error('native-bridge-seam-partial');
 if(digest===contract.applyDamageSHA256){
  if(!hasBridge||(text.match(/const thirdPartyIWR/g)??[]).length!==1)throw Error('native-bridge-seam-partial');
  const statement=bridgeStatement(text),profile=NATIVE_IWR_PROFILES[version],full=hash(source);
  if(profile&&[profile.patchedSHA256,profile.sharedManualCompositionSHA256].includes(full))return {status:'unchanged',buffer:Buffer.from(source),descriptor:{version,sourceSHA256:profile.originalSHA256,protocol:profile.protocol}};
  if(statement.includes('"sourceContract"')){
   const descriptor=automaticDescriptor(statement);
   if(descriptor.sourceContract!==contract.id||descriptor.methodSHA256!==contract.applyDamageSHA256||descriptor.protocol!==NATIVE_IWR_BRIDGE_PROTOCOL||!/^[a-f0-9]{64}$/.test(descriptor.sourceSHA256))throw Error('native-bridge-seam-descriptor');
   return {status:'unchanged',buffer:Buffer.from(source),descriptor};
  }
  const legacy=NATIVE_IWR_PROFILES[contract.shield==='O'?'8.5.1':'8.5.0'];
  const expected=`\tstatic thirdPartyNativeIWRBridge = Object.freeze({ version: "${legacy.systemVersion}", sourceSHA256: "${legacy.originalSHA256}", protocol: "${legacy.protocol}", applyDamage: this.prototype.applyDamage });`;
  if(statement!==expected)throw Error('native-bridge-seam-descriptor');
  const descriptor={version:String(version??''),sourceSHA256:hash(source),protocol:NATIVE_IWR_BRIDGE_PROTOCOL,sourceContract:contract.id,methodSHA256:contract.applyDamageSHA256};
  return {status:'patch',buffer:Buffer.from(once(text,statement,bridgeLine(descriptor))),descriptor};
 }
 const descriptor={version:String(version??''),sourceSHA256:hash(source),protocol:NATIVE_IWR_BRIDGE_PROTOCOL,sourceContract:contract.id,methodSHA256:contract.applyDamageSHA256};
 text=once(text,methodAnchor,bridgeLine(descriptor)+'\n'+methodAnchor);
 text=once(text,health,`\t\tconst thirdPartyIWR = game.modules.get(\`pf2e-third-party-automation\`)?.api${contract.callbackSignature};\n\t\tif (thirdPartyIWR && await thirdPartyIWR === true) x = ${contract.shield} + ie;\n`+health);
 if(hash(nativeMethod(text))!==contract.applyDamageSHA256)throw Error('native-bridge-seam-output');
 return {status:'patch',buffer:Buffer.from(text),descriptor};
}
function batchObserver(stack){
 let observer=read('native-manual-pool-batch/observer.js');
 observer=once(observer,"model:'numeric-empty-reception.v1'","model:'numeric-static-reception.v1',staticReceiverModelVersion:1,receiverPredicateModelVersion:1");
 const a=observer.indexOf(' function reception(actor){'),b=observer.indexOf(' function candidateCurrent(inv,candidate){',a);if(a<0||b<a)throw Error('native-shared-seam-reception');observer=observer.slice(0,a)+observer.slice(b);
 observer=once(observer,'  reception(candidate.contextualActor);',"  if(!candidate.receiver.isCurrent())fail('evidence-changed');");
 observer=once(observer,'    const captured={...candidate,targetOrdinal,','    const receiver=__nativeManualPoolStaticReceiver.model(candidate.contextualActor,candidate.params);\n    const captured={...candidate,receiver,targetOrdinal,');
 observer=once(observer,'contextualActor:candidate.contextualActor,\n    paramsSnapshot:','contextualActor:candidate.contextualActor,receiver:Object.freeze({qualified:true,amount:candidate.receiver.amount,flatTotal:candidate.receiver.flatTotal,entries:candidate.receiver.entries}),\n    paramsSnapshot:');
 observer=once(observer,'   // This model has equal amounts; the original target order decides the tie.\n   if(members[0].targetOrdinal!==entry.selectedOrdinal)fail(\'selection-mismatch\');',"   const largest=Math.max(...members.map(member=>member.receiver.amount));\n   if(members.find(member=>member.receiver.amount===largest).targetOrdinal!==entry.selectedOrdinal)fail('selection-mismatch');");
 observer=once(observer,'reception(candidate.contextualActor);return true','return candidate.receiver.isCurrent()');
 return 'const __nativeReceiverStacking=(()=>{\n'+stack+'return applyStackingRules;})();\n'+read('native-manual-pool-static-receiver/observer.js')+observer;
}
function setDescriptor(prefix,descriptor){return prefix.replace(/ const descriptor=Object.freeze\([^\n]+\);/,' const descriptor=Object.freeze('+JSON.stringify(descriptor)+');')}
const flatPush='(this.actor.synthetics.modifiers[r] ??= []).push(construct);',flatPatched=flatPush+' __nativeManualPoolStaticReceiver.register(construct,this,r,this.actor.synthetics.modifiers[r]);';
function instrumentBatch(original){
 const loop='\tfor (let n of gt(a, (e) => e.flags.pf2e.troop?.id ?? e)) {',call='\t\tawait a.applyDamage({';
 const l=original.indexOf(loop),c=original.indexOf(call),end=original.indexOf('\n\t\t});',c);if(!(l>=0&&c>l&&end>c))throw Error('native-shared-seam-batch');
 const preparation=original.slice(l+loop.length,c),params=original.slice(c+call.length,end);
 const added=`\tconst __batchTargets = [...gt(a, (e) => e.flags.pf2e.troop?.id ?? e)];\n\tconst __batch = __nativeManualPoolBatch.admit({message:e,roll:s,rollIndex:i,multiplier:t,addend:n,item:d,damage:c,targets:__batchTargets});\n\tif (__batch) {\n\t\tawait __nativeManualPoolBatch.run(__batch, async () => {\n\t\t\tconst candidates = [];\n\t\t\tfor (let n of __batchTargets) {${preparation}\t\tcandidates.push({token:n,patient:n.actor,contextualActor:a,params:{${params}\n\t\t}});\n\t\t\t}\n\t\t\treturn candidates;\n\t\t});\n\t\ttoggleOffShieldBlock(e.id);\n\t\treturn;\n\t}\n`;
 return original.slice(0,l)+added+original.slice(l).replace(loop,'\tfor (let n of __batchTargets) {');
}
const normalized=s=>s.replace(/ const descriptor=Object.freeze\([^\n]+\);/,' const descriptor=Object.freeze({});').replace("if(module?.version!=='3.56.5')return;","if(!module)return;");
function removeNativePair(pf){
 const observed=observerRegion(pf,'pf2e');pf=once(pf,observed.region,'');
 const expected=batchObserver(stackingRegion(pf));
 if(normalized(observed.region)!==normalized(expected))throw Error('native-shared-seam-observer');
 if(!observed.statement.includes('"sourceContract"')&&observed.statement!==expected.match(/ const descriptor=Object.freeze\([^\n]+\);/)?.[0])throw Error('native-shared-seam-descriptor');
 if(hash(batchRegion(pf))!==shape.batchPatched)throw Error('native-shared-seam-batch');
 let region=batchRegion(pf),start=region.indexOf('\tconst __batchTargets ='),end=region.indexOf('\n\tfor (let n of __batchTargets) {',start)+1;
 if(start<0||end<start)throw Error('native-shared-seam-batch');
 const original=region.slice(0,start)+region.slice(end).replace('\tfor (let n of __batchTargets) {','\tfor (let n of gt(a, (e) => e.flags.pf2e.troop?.id ?? e)) {');
 pf=once(pf,region,original);pf=once(pf,flatPatched,flatPush);pf=once(pf,'jm.onInit();__nativeManualPoolBatch.install();','jm.onInit();');
 return pf;
}
function removeToolPair(tb){
 const observed=observerRegion(tb,'toolbelt');
 const expected=read('toolbelt-manual-pool/observer.js').split('/* end toolbelt manual pool */')[0];
 if(normalized(observed.region)!==normalized(expected))throw Error('toolbelt-source-seam-observer');
 if(!observed.statement.includes('"sourceContract"')&&observed.statement!==expected.match(/ const descriptor=Object.freeze\([^\n]+\);/)?.[0])throw Error('toolbelt-source-seam-descriptor');
 tb=once(tb,observed.region+'/* end toolbelt manual pool */\n','','toolbelt-source-seam');tb=once(tb,TOOL_RECEIVE_PATCHED,TOOL_RECEIVE,'toolbelt-source-seam');return once(tb,TOOL_FORWARD_PATCHED,TOOL_FORWARD,'toolbelt-source-seam');
}
export function buildSharedManualPair({pf2eSource,toolbeltSource,pf2eVersion,toolbeltVersion}={}){
 let pf=sourceText(pf2eSource),tb=sourceText(toolbeltSource);
 const p=pf.includes('__nativeManualPoolBatch'),t=tb.includes('__toolbeltManualPool');
 if(!p&&(pf.includes('__nativeManualPoolStaticReceiver')||pf.includes('__nativeReceiverStacking')))throw Error('native-shared-seam-partial');
 let a,b;
 if(p){const header=observerRegion(pf,'pf2e').statement;if(header.includes('"sourceContract"'))a=automaticDescriptor(header);pf=removeNativePair(pf)}
 if(t){const header=observerRegion(tb,'toolbelt').statement;if(header.includes('"sourceContract"'))b=automaticDescriptor(header);tb=removeToolPair(tb)}
 if(hash(batchRegion(pf))!==shape.batchOriginal||hash(flatRegion(pf))!==shape.flatOriginal||hash(stackingRegion(pf))!==shape.stacking)throw Error('native-shared-seam-shape');
 const regions=toolRegions(tb);for(const [key,value]of Object.entries(regions))if(hash(value)!==shape['tool'+key[0].toUpperCase()+key.slice(1)])throw Error('toolbelt-source-seam-'+key);
 if(!tb.includes('r("_preUpdate",this.#h)'))throw Error('toolbelt-source-seam-wrapper');
 const native=buildNativeBridge({source:Buffer.from(pf),version:pf2eVersion});if(native.status!=='unchanged')throw Error('native-shared-seam-bridge-required');
 const installedBridge=buildNativeBridge({source:pf2eSource,version:pf2eVersion});
 const legacy=hash(pf2eSource)===legacyPF&&hash(toolbeltSource)===legacyTB&&pf2eVersion==='8.5.1'&&toolbeltVersion==='3.56.5';
 if(a&&!isAutomaticBatchDescriptor(a)||b&&!isAutomaticToolDescriptor(b))throw Error('native-shared-seam-descriptor');
 if(legacy||a&&b)return {pf2e:installedBridge.buffer,toolbelt:Buffer.from(toolbeltSource),alreadyPatched:installedBridge.status==='unchanged'};
 const pd={version:2,protocol:'pf2e-third-party-automation:manual-pool-batch:1',providerId:'pf2e',providerVersion:String(pf2eVersion??''),baseSourceSHA256:hash(pf2eSource),sourceContract:shape.id,model:'numeric-static-reception.v1',staticReceiverModelVersion:1,receiverPredicateModelVersion:1};
 const td={version:2,hpBaselineGuardVersion:1,providerId:'pf2e-toolbelt',providerVersion:String(toolbeltVersion??''),sourceSHA256:hash(toolbeltSource),sourceContract:shape.toolbeltId};
 const original=batchRegion(pf),patched=instrumentBatch(original);if(hash(patched)!==shape.batchPatched)throw Error('native-shared-seam-output');
 pf=once(pf,original,setDescriptor(batchObserver(stackingRegion(pf)),pd)+patched);pf=once(pf,flatPush,flatPatched);pf=once(pf,'jm.onInit();','jm.onInit();__nativeManualPoolBatch.install();');
 tb=once(tb,TOOL_RECEIVE,TOOL_RECEIVE_PATCHED,'toolbelt-source-seam');tb=once(tb,TOOL_FORWARD,TOOL_FORWARD_PATCHED,'toolbelt-source-seam');
 const observer=setDescriptor(read('toolbelt-manual-pool/observer.js'),td).replace("if(module?.version!=='3.56.5')return;","if(!module)return;");
 const complete=buildNativeBridge({source:Buffer.from(pf),version:pf2eVersion});
 const toolOutput=tb.startsWith('\uFEFF')?'\uFEFF'+observer+tb.slice(1):observer+tb;
 return {pf2e:a?installedBridge.buffer:complete.buffer,toolbelt:b?Buffer.from(toolbeltSource):Buffer.from(toolOutput),alreadyPatched:false,descriptors:{pf2e:a??pd,toolbelt:b??td}};
}
