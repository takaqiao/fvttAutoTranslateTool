import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash,randomUUID} from 'node:crypto';
import {patchStaticReceiver} from './native-manual-pool-static-receiver/patch.mjs';
import {patchToolbeltManualPool} from './toolbelt-manual-pool/patch.mjs';

const moduleRoot=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),checkout=path.resolve(moduleRoot,'../..'),protocol='manual-shared-maintenance.v1';
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const targets=[
 {id:'pf2e',version:'8.5.1',manifest:'systems/pf2e/system.json',relativePath:'systems/pf2e/pf2e.mjs',backup:'original-pf2e.mjs',candidate:'patched-pf2e.mjs',originalSHA256:'d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157',outputSHA256:'9be357c96dff3d0790edcb0d7889db98cfded0f41f34fd161233ea0bdd0dab11'},
 {id:'pf2e-toolbelt',version:'3.56.5',manifest:'modules/pf2e-toolbelt/module.json',relativePath:'modules/pf2e-toolbelt/scripts/main.js',backup:'original-main.js',candidate:'patched-main.js',originalSHA256:'2946fa27eaf0963098f9f48ee48f777c5cf65166b56b043de9404f09813c240f',outputSHA256:'6937743860c8b2fbeef1037bb99792307cb3fbe7a2540ae0b1ac7010a16a5d90'}
];
const toolFiles=['tools/manual-shared-maintenance.mjs','tools/native-manual-pool-static-receiver/patch.mjs','tools/native-manual-pool-static-receiver/observer.js','tools/toolbelt-manual-pool/patch.mjs','tools/toolbelt-manual-pool/observer.js','tools/native-manual-pool-batch/observer.js','tools/native-manual-pool-batch/patch.mjs','tools/native-manual-pool-batch/seam.test.mjs','tools/native-manual-pool-batch/README.md','tests/exploration/manual-pool-batch.test.mjs'];
const normalize=p=>process.platform==='win32'?p.toLowerCase():p,inside=(p,root)=>normalize(p)===normalize(root)||normalize(p).startsWith(normalize(root)+path.sep);
function noLinks(file){for(let at=path.resolve(file);;at=path.dirname(at)){if(fs.existsSync(at)&&fs.lstatSync(at).isSymbolicLink())throw Error('symbolic-path-rejected');if(path.dirname(at)===at)break;}}
function regular(file){noLinks(file);if(!fs.lstatSync(file).isFile())throw Error('regular-file-required');return fs.readFileSync(file);}
function dataDirectory(dataRoot){if(typeof dataRoot!=='string'||!dataRoot)throw Error('data-root-required');noLinks(dataRoot);if(!fs.statSync(dataRoot).isDirectory())throw Error('data-root-required');return fs.realpathSync(dataRoot);}
function privateDirectory(dataRoot,planDir,fresh){
 if(typeof planDir!=='string'||!planDir)throw Error('private-new-plan-required');const resolved=path.resolve(planDir);noLinks(resolved);
 if(inside(resolved,dataRoot)||inside(resolved,checkout)||fresh&&fs.existsSync(resolved))throw Error('private-new-plan-required');
 const parent=fs.realpathSync(path.dirname(resolved)),canonical=path.join(parent,path.basename(resolved));
 if(inside(canonical,dataRoot)||inside(canonical,fs.realpathSync(checkout)))throw Error('private-new-plan-required');
 if(!fresh&&!fs.statSync(canonical).isDirectory())throw Error('plan-directory-required');return canonical;
}
function versions(dataRoot){for(const target of targets){const bytes=regular(path.join(dataRoot,target.manifest));if(bytes.length>1024*1024)throw Error('version-manifest-too-large');const manifest=JSON.parse(bytes);if(manifest.id!==target.id||manifest.version!==target.version)throw Error('version-mismatch:'+target.id);}}
function sourceState(dataRoot){return targets.map(target=>{const sha256=hash(regular(path.join(dataRoot,target.relativePath)));if(sha256!==target.originalSHA256&&sha256!==target.outputSHA256)throw Error('source-mismatch:'+target.id);return {id:target.id,sha256,state:sha256===target.originalSHA256?'original':'installed'};});}
const pair=states=>states.every(s=>s.state==='original')?'original':states.every(s=>s.state==='installed')?'installed':'partial';
function planFor(dataRoot,planDir){return {protocol,dataRoot,planDir,targets:targets.map(t=>({...t})),tools:toolFiles.map(relativePath=>({relativePath,sha256:hash(regular(path.join(moduleRoot,relativePath)))}))};}
function generated(originals){const candidates=[patchStaticReceiver(originals[0]).bytes,patchToolbeltManualPool(originals[1]).bytes];for(let i=0;i<targets.length;i++)if(hash(candidates[i])!==targets[i].outputSHA256)throw Error('generated-output-mismatch:'+targets[i].id);return candidates;}
function checkedPlan(dataRoot,planDir){
 const bytes=regular(path.join(planDir,'plan.json'));if(bytes.length>65536)throw Error('plan-mismatch');const plan=JSON.parse(bytes);
 if(JSON.stringify(plan)!==JSON.stringify(planFor(dataRoot,planDir)))throw Error('plan-mismatch');
 const originals=targets.map(t=>{const bytes=regular(path.join(planDir,t.backup));if(hash(bytes)!==t.originalSHA256)throw Error('backup-mismatch:'+t.id);return bytes;});
 const candidates=generated(originals);for(let i=0;i<targets.length;i++){const saved=regular(path.join(planDir,targets[i].candidate));if(hash(saved)!==targets[i].outputSHA256||!saved.equals(candidates[i]))throw Error('candidate-mismatch:'+targets[i].id);}
 return {originals,candidates};
}
export function readManualSharedStatus({dataRoot}={}){const root=dataDirectory(dataRoot);versions(root);const states=sourceState(root);return {protocol,dataRoot:root,pair:pair(states),targets:states};}
export function prepareManualShared({dataRoot,planDir}={}){
 const root=dataDirectory(dataRoot),privateRoot=privateDirectory(root,planDir,true);versions(root);const states=sourceState(root);if(pair(states)!=='original')throw Error('prepare-original-pair-required');
 const originals=targets.map(t=>regular(path.join(root,t.relativePath))),candidates=generated(originals),plan=planFor(root,privateRoot);
 // Complete validation precedes creating the private evidence directory.
 fs.mkdirSync(privateRoot,{mode:0o700});for(let i=0;i<targets.length;i++){fs.writeFileSync(path.join(privateRoot,targets[i].backup),originals[i],{flag:'wx'});fs.writeFileSync(path.join(privateRoot,targets[i].candidate),candidates[i],{flag:'wx'});}fs.writeFileSync(path.join(privateRoot,'plan.json'),JSON.stringify(plan,null,2)+'\n',{flag:'wx'});return plan;
}
function changePair(action,{dataRoot,planDir,operatorAssertion}={}){
 if(operatorAssertion!=='foundry-and-clients-stopped')throw Error('operator-assertion-required');
 const root=dataDirectory(dataRoot),privateRoot=privateDirectory(root,planDir,false);versions(root);const {originals,candidates}=checkedPlan(root,privateRoot),before=sourceState(root),desired=action==='apply'?candidates:originals,desiredState=action==='apply'?'installed':'original';
 const attemptId=randomUUID(),attempt={protocol,action,operatorAssertion,dataRoot:root,planDir:privateRoot,before,attemptId};
 fs.writeFileSync(path.join(privateRoot,'attempt-'+attemptId+'.json'),JSON.stringify(attempt,null,2)+'\n',{flag:'wx'});let writes=0;
 try{
  for(let i=0;i<targets.length;i++){
   const target=targets[i],file=path.join(root,target.relativePath),current=hash(regular(file));if(current!==before[i].sha256)throw Error('source-changed-during-maintenance:'+target.id);
   if(before[i].state===desiredState)continue;
   fs.writeFileSync(file,desired[i]);writes++;if(hash(regular(file))!==hash(desired[i]))throw Error('write-readback-mismatch:'+target.id);
  }
  versions(root);const after=sourceState(root);if(pair(after)!==desiredState)throw Error('pair-readback-mismatch');const result={...attempt,pair:pair(after),writes,after,status:'completed'};
  fs.writeFileSync(path.join(privateRoot,'result-'+attemptId+'.json'),JSON.stringify(result,null,2)+'\n',{flag:'wx'});return result;
 }catch(error){
  // Keep saved originals, candidates and the start record. Never silently undo
  // a partial write or replace an edit whose source is now unknown.
  const failed={...attempt,status:'uncertain',writes,error:String(error.message??error),current:targets.map(t=>{try{return {id:t.id,sha256:hash(regular(path.join(root,t.relativePath)))}}catch{return {id:t.id,unreadable:true}}})};
  try{fs.writeFileSync(path.join(privateRoot,'failed-'+attemptId+'.json'),JSON.stringify(failed,null,2)+'\n',{flag:'wx'});}catch{}throw error;
 }
}
export const applyManualShared=options=>changePair('apply',options);
export const restoreManualShared=options=>changePair('restore',options);
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 try{
  const [action,...args]=process.argv.slice(2),options={},names={'--data-root':'dataRoot','--plan-dir':'planDir','--operator-assertion':'operatorAssertion'};
  if(!['prepare','status','apply','restore'].includes(action)||args.length%2)throw Error('usage: prepare|status|apply|restore --data-root <Data> [--plan-dir <private-directory>] [--operator-assertion foundry-and-clients-stopped]');
  for(let i=0;i<args.length;i+=2){const name=names[args[i]];if(!name||Object.hasOwn(options,name)||!args[i+1])throw Error('invalid-arguments');options[name]=args[i+1];}
  const result=({prepare:prepareManualShared,status:readManualSharedStatus,apply:applyManualShared,restore:restoreManualShared})[action](options);console.log(JSON.stringify(result));
 }catch(error){console.error(JSON.stringify({status:'rejected',reason:String(error.message??error)}));process.exitCode=1;}
}
