import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';

export const BASE_SHA='d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157';
const directory=path.dirname(fileURLToPath(import.meta.url));
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const start='async function applyDamageFromMessage({ message: e, multiplier: t = 1, addend: n = 0, promptModifier: r = !1, rollIndex: i = 0 }) {';
const end='async function shiftAdjustDamage(e, t, n) {';
const loop='\tfor (let n of gt(a, (e) => e.flags.pf2e.troop?.id ?? e)) {';
const call='\t\tawait a.applyDamage({';
const init='jm.onInit();';
function once(source,anchor){if(source.split(anchor).length!==2)throw Error('native-manual-batch-anchor-not-unique');return source.indexOf(anchor)}
function instrumentNativeManualPoolBatch(source,observer){
 const from=once(source,start),to=once(source,end),loopFrom=once(source,loop),callFrom=once(source,call),initFrom=once(source,init);
 if(!(from<loopFrom&&loopFrom<callFrom&&callFrom<to))throw Error('native-manual-batch-source-region-unavailable');
 const original=source.slice(from,to),localLoop=original.indexOf(loop),localCall=original.indexOf(call),callEnd=original.indexOf('\n\t\t});',localCall);
 if(callEnd<localCall)throw Error('native-manual-batch-source-call-unavailable');
 const preparation=original.slice(localLoop+loop.length,localCall),params=original.slice(localCall+call.length,callEnd);
 const participating=`\tconst __batchTargets = [...gt(a, (e) => e.flags.pf2e.troop?.id ?? e)];\n\tconst __batch = __nativeManualPoolBatch.admit({message:e,roll:s,rollIndex:i,multiplier:t,addend:n,item:d,damage:c,targets:__batchTargets});\n\tif (__batch) {\n\t\tawait __nativeManualPoolBatch.run(__batch, async () => {\n\t\t\tconst candidates = [];\n\t\t\tfor (let n of __batchTargets) {${preparation}\t\tcandidates.push({token:n,patient:n.actor,contextualActor:a,params:{${params}\n\t\t}});\n\t\t\t}\n\t\t\treturn candidates;\n\t\t});\n\t\ttoggleOffShieldBlock(e.id);\n\t\treturn;\n\t}\n`;
 const patched=original.slice(0,localLoop)+participating+original.slice(localLoop).replace(loop,'\tfor (let n of __batchTargets) {');
 let output=source.slice(0,from)+observer+patched+source.slice(to);
 output=output.replace(init,init+'__nativeManualPoolBatch.install();');
 return {bytes:Buffer.from(output),originalRegion:original,patchedRegion:patched,region:{startUTF16:from,endUTF16Exclusive:to,initUTF16:initFrom}};
}
export function patchNativeManualPoolBatch(bytes){
 const inputSHA=hash(bytes);if(inputSHA!==BASE_SHA)throw Error('native-manual-batch-source-sha-mismatch');
 const observer=fs.readFileSync(path.join(directory,'observer.js'),'utf8');
 return {...instrumentNativeManualPoolBatch(Buffer.from(bytes).toString('utf8'),observer),inputSHA,observerSHA:hash(observer)};
}
export function prepareNativeManualPoolBatch(sourcePath,outputDirectory){
 const output=path.resolve(outputDirectory),workspace=path.resolve(directory,'../../../..');
 const normalized=value=>process.platform==='win32'?value.toLowerCase():value;
 if(normalized(output)===normalized(workspace)||normalized(output).startsWith(normalized(workspace)+path.sep))throw Error('native-manual-batch-output-in-workspace');
 if(fs.existsSync(output))throw Error('native-manual-batch-output-already-exists');
 const result=patchNativeManualPoolBatch(fs.readFileSync(sourcePath));fs.mkdirSync(output,{recursive:true});
 fs.writeFileSync(path.join(output,'patched-pf2e.mjs'),result.bytes,{flag:'wx'});
 const manifest={protocol:'pf2e-native-manual-pool-batch.v1',status:'prepared-not-installed',source:{file:path.resolve(sourcePath),sha256:result.inputSHA},outputSHA256:hash(result.bytes),observerSHA256:result.observerSHA,
  patcherSHA256:hash(fs.readFileSync(fileURLToPath(import.meta.url))),region:{...result.region,originalSHA256:hash(result.originalRegion),patchedSHA256:hash(result.patchedRegion)}};
 fs.writeFileSync(path.join(output,'manifest.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});return manifest;
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const [sourcePath,outputDirectory]=process.argv.slice(2);if(!sourcePath||!outputDirectory)throw Error('usage: patch fixed-pf2e-source private-new-output-directory');
 console.log(JSON.stringify(prepareNativeManualPoolBatch(sourcePath,outputDirectory)));
}
