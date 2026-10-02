import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';

export const BASE_SHA='2946fa27eaf0963098f9f48ee48f777c5cf65166b56b043de9404f09813c240f';
const directory=path.dirname(fileURLToPath(import.meta.url));
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
export function patchToolbeltManualPool(bytes){
 if(hash(bytes)!==BASE_SHA)throw Error('toolbelt-manual-pool-source-mismatch');
 let source=Buffer.from(bytes).toString('utf8');
 const replacements=[
  ['#g({master:e,...n}){this.isValidMaster(e)&&e.update(n)}','#g({master:e,__explorationManualPool:x,...n},s){return __toolbeltManualPool.receive(e,n,x,s,()=>this.isValidMaster(e)&&e.update(n))}'],
  ['m.isOwner?m.update(g):this.#t.emit({master:m,...g})','let x=__toolbeltManualPool.forward(e,m,g,r,()=>m.update(g),p=>this.#t.emit(p));if(x)await x']
 ];
 for(const [before,after]of replacements){if(source.split(before).length!==2)throw Error('toolbelt-manual-pool-anchor-mismatch');source=source.replace(before,after)}
 const observer=fs.readFileSync(path.join(directory,'observer.js'),'utf8');return {bytes:Buffer.from(observer+source)};
}
export function prepareToolbeltManualPool(sourcePath,outputDirectory){
 const output=path.resolve(outputDirectory),workspace=path.resolve(directory,'../../../..');
 if(output.toLowerCase()===workspace.toLowerCase()||output.toLowerCase().startsWith(workspace.toLowerCase()+path.sep)||fs.existsSync(output))throw Error('toolbelt-private-output-required');
 const source=fs.readFileSync(sourcePath),result=patchToolbeltManualPool(source);fs.mkdirSync(output,{recursive:true});
 fs.writeFileSync(path.join(output,'original-main.js'),source,{flag:'wx'});fs.writeFileSync(path.join(output,'patched-main.js'),result.bytes,{flag:'wx'});
 const inventory={status:'prepared-not-installed',inputSHA256:hash(source),outputSHA256:hash(result.bytes),observerSHA256:hash(fs.readFileSync(path.join(directory,'observer.js'))),patcherSHA256:hash(fs.readFileSync(fileURLToPath(import.meta.url)))};
 fs.writeFileSync(path.join(output,'manifest.json'),JSON.stringify(inventory,null,2)+'\n',{flag:'wx'});return inventory;
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const [source,output]=process.argv.slice(2);if(!source||!output)throw Error('usage: patch source private-output-directory');console.log(JSON.stringify(prepareToolbeltManualPool(source,output)));
}
