import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';

const directory=path.dirname(fileURLToPath(import.meta.url)),root=path.resolve(directory,'../..');
const hash=value=>createHash('sha256').update(value).digest('hex');
const componentFiles={
 'tools/native-manual-pool-batch/observer.js':'476a7251030f10673661e89a4f76dd89a1873e6e1c2c5e72d719f787b2df9665',
 'tools/native-manual-pool-batch/patch.mjs':'401bd770e4266bdb917c952e1b8475f7a27d3c4188964e23f9c535f9cff709d7',
 'tools/native-manual-pool-batch/seam.test.mjs':'7fb54e27710d813bac444f3c614fac1332afc226c3ee08aebb5a8cbd1008a748',
 'tools/native-manual-pool-batch/README.md':'84df7c573135e0297bd53e6463450b514905f18e9a11c85843174edfd6961efd',
 'tests/exploration/manual-pool-batch.test.mjs':'b865292a6034c60ef5cac80861b32257f4af5933f7bb16e1b9d407e95a1d7940'
};
const BASE_SHA='d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157';
function verifyComponents(){for(const [file,sha]of Object.entries(componentFiles))if(hash(fs.readFileSync(path.join(root,file)))!==sha)throw Error('static-receiver-component-mismatch')}
verifyComponents();
const {patchNativeManualPoolBatch}=await import('../native-manual-pool-batch/patch.mjs');
export const BATCH_SHA='9c8e5f66313e49786b08826a0e1a60dd8605743c523808f7ca2aa941e325f64a';
function replace(source,before,after){if(source.split(before).length!==2)throw Error('static-receiver-anchor-mismatch');return source.replace(before,after)}
export function patchStaticReceiver(bytes){
 if(hash(bytes)!==BASE_SHA)throw Error('static-receiver-source-mismatch');
 verifyComponents();
 const batch=patchNativeManualPoolBatch(bytes);if(hash(batch.bytes)!==BATCH_SHA)throw Error('static-receiver-component-output-mismatch');
 const fixed=Buffer.from(bytes).toString('utf8');let source=batch.bytes.toString('utf8');
 const from=fixed.indexOf('var HIGHER_BONUS ='),to=fixed.indexOf('var StatisticModifier =',from),stack=fixed.slice(from,to);
 if(hash(stack)!=='c47f085d0b258d7ec4dadb37dfa7d68b17ff8e51d31af7173dc4605a99a37cba')throw Error('static-receiver-stacking-source-mismatch');
 source=replace(source,'(this.actor.synthetics.modifiers[r] ??= []).push(construct);','(this.actor.synthetics.modifiers[r] ??= []).push(construct); __nativeManualPoolStaticReceiver.register(construct,this,r,this.actor.synthetics.modifiers[r]);');
 const observer=fs.readFileSync(path.join(directory,'observer.js'),'utf8');
 source=replace(source,'const __nativeManualPoolBatch=', 'const __nativeReceiverStacking=(()=>{\n'+stack+'return applyStackingRules;})();\n'+observer+'const __nativeManualPoolBatch=');
 source=replace(source,"model:'numeric-empty-reception.v1'","model:'numeric-static-reception.v1',staticReceiverModelVersion:1,receiverPredicateModelVersion:1");
 const a=source.indexOf(' function reception(actor){'),b=source.indexOf(' function candidateCurrent(inv,candidate){',a);if(a<0||b<a)throw Error('static-receiver-reception-anchor');
 source=source.slice(0,a)+source.slice(b);
 source=replace(source,'  reception(candidate.contextualActor);','  if(!candidate.receiver.isCurrent())fail(\'evidence-changed\');');
 source=replace(source,'    const captured={...candidate,targetOrdinal,','    const receiver=__nativeManualPoolStaticReceiver.model(candidate.contextualActor,candidate.params);\n    const captured={...candidate,receiver,targetOrdinal,');
 source=replace(source,'contextualActor:candidate.contextualActor,\n    paramsSnapshot:','contextualActor:candidate.contextualActor,receiver:Object.freeze({qualified:true,amount:candidate.receiver.amount,flatTotal:candidate.receiver.flatTotal,entries:candidate.receiver.entries}),\n    paramsSnapshot:');
 source=replace(source,'   // This model has equal amounts; the original target order decides the tie.\n   if(members[0].targetOrdinal!==entry.selectedOrdinal)fail(\'selection-mismatch\');',"   const largest=Math.max(...members.map(member=>member.receiver.amount));\n   if(members.find(member=>member.receiver.amount===largest).targetOrdinal!==entry.selectedOrdinal)fail('selection-mismatch');");
 source=replace(source,'reception(candidate.contextualActor);return true','return candidate.receiver.isCurrent()');
 return {bytes:Buffer.from(source),inputSHA:BASE_SHA,batchSHA:BATCH_SHA,observerSHA:hash(observer),stackingSHA:hash(stack),componentFiles};
}
export function prepareStaticReceiver(sourcePath,outputDirectory){
 const output=path.resolve(outputDirectory),repo=path.resolve(root,'../..'),normalize=value=>process.platform==='win32'?value.toLowerCase():value;
 if(normalize(output)===normalize(repo)||normalize(output).startsWith(normalize(repo)+path.sep)||fs.existsSync(output))throw Error('static-receiver-private-new-output-required');
 const result=patchStaticReceiver(fs.readFileSync(sourcePath));fs.mkdirSync(output,{recursive:true});fs.writeFileSync(path.join(output,'patched-pf2e.mjs'),result.bytes,{flag:'wx'});
 const manifest={status:'prepared-not-installed',protocol:'native-manual-pool-static-receiver.v1',input:{file:path.resolve(sourcePath),sha256:result.inputSHA},batchSHA:result.batchSHA,observerSHA:result.observerSHA,stackingSHA:result.stackingSHA,componentFiles,outputSHA:hash(result.bytes),patcherSHA:hash(fs.readFileSync(fileURLToPath(import.meta.url)))};
 fs.writeFileSync(path.join(output,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});return manifest;
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){const [source,output]=process.argv.slice(2);if(!source||!output)throw Error('usage: patch fixed-source private-new-output');console.log(JSON.stringify(prepareStaticReceiver(source,output)))}
