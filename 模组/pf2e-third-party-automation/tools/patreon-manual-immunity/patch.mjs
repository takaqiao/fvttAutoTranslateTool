import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';

export const BASE_SHA='89ded325b92fa6b03dcf9257337ae2d628b3e99c987fe22cef9fff4e1837f4e9';
export const TIME_V5_SHA='d29a3878f9521cc185eb295288fe660b4c41a86a0a07fcc966423e887b15de22';
const directory=path.dirname(fileURLToPath(import.meta.url));
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
function once(source,before,after){if(source.split(before).length!==2)throw Error('manual-immunity-patch-anchor-not-unique');return source.replace(before,after)}
export function patchPatreonManualImmunity(bytes,{patchTime}={}){
 const inputSHA=hash(bytes);let timeBytes=bytes;
 if(inputSHA===BASE_SHA){if(typeof patchTime!=='function')throw Error('manual-immunity-time-patcher-required');timeBytes=patchTime(bytes).bytes}
 else if(inputSHA!==TIME_V5_SHA)throw Error('manual-immunity-source-sha-mismatch');
 if(hash(timeBytes)!==TIME_V5_SHA)throw Error('manual-immunity-time-source-sha-mismatch');
 let source=Buffer.from(timeBytes).toString('utf8');
 const maStart=source.indexOf('async function Ma('),maEnd=source.indexOf('r(Ma,"treatWounds");',maStart)+'r(Ma,"treatWounds");'.length;
 if(maStart<0||maEnd<=maStart)throw Error('manual-immunity-source-region-unavailable');
 const original=source.slice(maStart,maEnd);
 let patched=once(original,'async function Ma(a,e){','async function Ma(a,e){const x=__patreonManualImmunity.bind(a,e);');
 patched=once(patched,'await p(e,t),','await p(e,await __patreonManualImmunity.mark(t,x)),');
 const observer=fs.readFileSync(path.join(directory,'observer.js'),'utf8');
 source=source.slice(0,maStart)+observer+patched+source.slice(maEnd);
 source=once(source,'let i=await a.createEmbeddedDocuments("Item",[e]);','let i=await __patreonManualImmunity.create(a,e,()=>a.createEmbeddedDocuments("Item",[e]));');
 return {bytes:Buffer.from(source),inputSHA,timeSHA:hash(timeBytes),originalRegion:original,patchedRegion:patched,observerSHA:hash(observer)};
}
export function preparePatreonManualImmunity(sourcePath,outputDirectory,{patchTime}={}){
 const output=path.resolve(outputDirectory),workspace=path.resolve(directory,'../../../..');
 const destination=process.platform==='win32'?output.toLowerCase():output,checkout=process.platform==='win32'?workspace.toLowerCase():workspace;
 if(destination===checkout||destination.startsWith(checkout+path.sep))throw Error('manual-immunity-paid-output-in-workspace');
 if(fs.existsSync(output))throw Error('manual-immunity-output-already-exists');
 const bytes=fs.readFileSync(sourcePath),result=patchPatreonManualImmunity(bytes,{patchTime});
 fs.mkdirSync(output,{recursive:true});fs.writeFileSync(path.join(output,'input-index.js'),bytes,{flag:'wx'});fs.writeFileSync(path.join(output,'patched-index.js'),result.bytes,{flag:'wx'});
 const inventory={protocol:'patreon-native-manual-immunity.v1',status:'prepared-not-installed',source:{file:path.resolve(sourcePath),sha256:result.inputSHA},timeSHA256:result.timeSHA,
  outputSHA256:hash(result.bytes),observerSHA256:result.observerSHA,patcherSHA256:hash(fs.readFileSync(fileURLToPath(import.meta.url))),region:{originalSHA256:hash(result.originalRegion),patchedSHA256:hash(result.patchedRegion)}};
 fs.writeFileSync(path.join(output,'manifest.json'),JSON.stringify(inventory,null,2)+'\n',{flag:'wx'});return inventory;
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const [sourcePath,outputDirectory,timePatcherPath]=process.argv.slice(2);if(!sourcePath||!outputDirectory)throw Error('usage: patch source private-output-directory [existing-v5-patcher]');
 const patchTime=timePatcherPath?(await import(pathToFileURL(path.resolve(timePatcherPath)).href)).patchPatreon:undefined;
 console.log(JSON.stringify(preparePatreonManualImmunity(sourcePath,outputDirectory,{patchTime})));
}
