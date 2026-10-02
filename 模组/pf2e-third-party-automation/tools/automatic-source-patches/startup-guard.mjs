import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
const posix=path.posix;
export function startTicks(stat){const tail=String(stat).slice(String(stat).lastIndexOf(')')+1).trim().split(/\s+/);assert(/^\d+$/.test(tail[19]??''),'Malformed proc stat');return tail[19];}
export function dataPathArgument(argv){let value=null;for(let i=0;i<argv.length;i++){const arg=argv[i];if(arg==='--dataPath'||arg.startsWith('--dataPath=')){assert.equal(value,null,'Duplicate dataPath');value=arg==='--dataPath'?argv[++i]:arg.slice(11);assert(value&&!value.startsWith('--'),'Missing dataPath value');}}return value;}
export function processMetadata(bytes){const allowed=new Set(['pm_exec_path','pm_cwd','args','pm_id','name']),result={};for(const entry of Buffer.from(bytes).toString('utf8').split('\0')){const equal=entry.indexOf('=');if(equal<0)continue;const key=entry.slice(0,equal);if(allowed.has(key)){assert(!Object.hasOwn(result,key),'Duplicate PM2 process metadata');result[key]=entry.slice(equal+1);}}return result;}
function majorMinor(dev){dev=BigInt(dev);return{major:((dev>>8n)&0xfffn)|((dev>>32n)&0xfffff000n),minor:(dev&0xffn)|((dev>>12n)&0xffffff00n)};}
export function matchesLock(text,stat,pid){const{major,minor}=majorMinor(stat.dev);return String(text).split('\n').some(line=>{const m=/^lock:\s+\d+:\s+FLOCK\s+ADVISORY\s+WRITE\s+(\d+)\s+([a-f\d]+):([a-f\d]+):(\d+)\s+0\s+EOF\s*$/i.exec(line);return m&&Number(m[1])===pid&&BigInt('0x'+m[2])===major&&BigInt('0x'+m[3])===minor&&BigInt(m[4])===BigInt(stat.ino);});}
const inTree=(root,p)=>p===root||p.startsWith(root+'/');
const sameFile=(a,b)=>a.dev===b.dev&&a.ino===b.ino;
export async function verifyHeldLock({lockPath,pid=process.pid,io=fs,procRoot='/proc'}){
 assert(posix.isAbsolute(lockPath));assert.equal(await io.realpath(lockPath),lockPath,'Lock path must be canonical');const lock=await io.lstat(lockPath,{bigint:true});assert(lock.isFile()&&!lock.isSymbolicLink()&&lock.nlink===1n,'Unsafe startup lock file');assert.equal(Number(lock.mode&0o22n),0,'Startup lock is writable by another user');
 const fds=await io.readdir(`${procRoot}/${pid}/fd`);for(const fd of fds.filter(x=>/^\d+$/.test(x))){let s;try{s=await io.stat(`${procRoot}/${pid}/fd/${fd}`,{bigint:true});}catch(e){if(e.code==='ENOENT')continue;throw e;}if(!sameFile(s,lock))continue;const info=await io.readFile(`${procRoot}/${pid}/fdinfo/${fd}`,'utf8');if(matchesLock(info,lock,pid))return {pid,fd:Number(fd),device:String(lock.dev),inode:String(lock.ino),lockPath};}
 throw Error('Startup process does not own the exact exclusive flock');
}
async function scanOnce({context,pid=process.pid,mainPath,io=fs,procRoot='/proc',signal}){
 const occupants=[],initial=(await io.readdir(procRoot)).filter(x=>/^\d+$/.test(x)),files=await Promise.all([context.bundlePath,context.manifestPath,context.systemDir].map(p=>io.stat(p,{bigint:true})));
 for(const id of initial){if(Number(id)===pid)continue;signal?.throwIfAborted();const prefix=`${procRoot}/${id}`;let initialStat;
  try{initialStat=await io.readFile(prefix+'/stat','utf8');}catch(e){if(e.code==='ENOENT')continue;throw e;}
  const ticks=startTicks(initialStat);if(String(initialStat).slice(String(initialStat).lastIndexOf(')')+1).trim().startsWith('Z '))continue;
  try{
   const argv=Buffer.from(await io.readFile(prefix+'/cmdline')).toString('utf8').split('\0').filter(Boolean),metadata=processMetadata(await io.readFile(prefix+'/environ'));
   if(!argv.length&&!Object.keys(metadata).length){assert.equal(startTicks(await io.readFile(prefix+'/stat','utf8')),ticks,'PID reused during guard scan');continue;}
   const cwd=await io.readlink(prefix+'/cwd');
   const foundry=metadata.pm_exec_path===mainPath||metadata.pm_exec_path?.endsWith('/main.mjs')||argv.some(arg=>posix.resolve(cwd,arg)===mainPath||/^node\s+\/.*\/main\.mjs$/.test(arg)||arg===`node ${mainPath}`);
   const directDataPath=dataPathArgument(argv),environmentDataPath=metadata.args?dataPathArgument(metadata.args.split(',')):null;
   if(directDataPath&&environmentDataPath)assert.equal(posix.resolve(cwd,directDataPath),posix.resolve(metadata.pm_cwd??cwd,environmentDataPath),'Conflicting process dataPath metadata');
   let dataPath=directDataPath??environmentDataPath,reason=null;
   if(dataPath){const supplied=posix.resolve(directDataPath?cwd:(metadata.pm_cwd??cwd),dataPath);let canonical;try{canonical=await io.realpath(supplied);}catch(e){if(e.code==='ENOENT'){if(supplied===context.dataPath)reason='unresolved-target-data-path';}else throw e;}
    if(canonical===context.dataPath)reason='same-data-path';
    if(!canonical&&foundry)reason??='foundry-data-path-unresolved';
    if(canonical&&!reason){try{const sys=await io.realpath(posix.join(canonical,'Data/systems/pf2e'));if(sys===context.systemDir)reason='same-system-directory';}catch(e){if(e.code!=='ENOENT')throw e;else if(foundry)reason='foundry-system-path-unresolved';}}
   }else if(foundry)reason='foundry-without-explicit-data-path';
   const busyTrees=[context.systemDir,...['Data','Logs','Config'].map(name=>posix.join(context.dataPath,name))];
   const fds=await io.readdir(prefix+'/fd');for(const fd of fds.filter(x=>/^\d+$/.test(x))){try{const target=await io.readlink(prefix+'/fd/'+fd);if(busyTrees.some(root=>inTree(root,target.replace(/ \(deleted\)$/,'')))){reason??='open-installation-file';break;}const st=await io.stat(prefix+'/fd/'+fd,{bigint:true});if(files.some(s=>sameFile(s,st))){reason??='same-system-inode';break;}}catch(e){if(e.code!=='ENOENT')throw e;}}
   assert.equal(startTicks(await io.readFile(prefix+'/stat','utf8')),ticks,'PID reused during guard scan');if(reason)occupants.push({pid:Number(id),startTicks:ticks,reason});
  }catch(e){if(e.code==='ENOENT'){try{await io.readFile(prefix+'/stat');}catch(gone){if(gone.code==='ENOENT')continue;} }throw e;}
 }
 const after=(await io.readdir(procRoot)).filter(x=>/^\d+$/.test(x));
 return{idle:occupants.length===0,dataPath:context.dataPath,occupants,scanned:initial.length,processSetChanged:!after.every(id=>initial.includes(id)||Number(id)===pid)};
}
export async function scanProcesses(options){
 // A short-lived unrelated process may appear during a full scan. Rescan from scratch;
 // never discard positive occupancy evidence and never call a changing set idle.
 for(let scanAttempts=1;scanAttempts<=3;scanAttempts++){options.signal?.throwIfAborted();const result=await scanOnce(options);if(!result.idle||!result.processSetChanged)return{...result,scanAttempts};}
 throw Object.assign(Error('Process set remains unstable after three complete scans; retry only in a controlled startup window'),{code:'process-set-unstable',retryable:true});
}
export function createServiceGuard({config,pid=process.pid,io=fs,procRoot='/proc'}){return async context=>{
 assert.equal(context.dataPath,config.dataPath);assert.equal(context.dataDirectory,posix.join(config.dataPath,'Data'));assert.equal(context.systemDir,posix.join(config.dataPath,'Data/systems/pf2e'));
 const lock=await verifyHeldLock({lockPath:config.lockPath,pid,io,procRoot});const result=await scanProcesses({context,pid,mainPath:config.mainPath,io,procRoot,signal:context.signal});return{...result,lock};
};}
