/** Merge observational maintenance only; actual actions keep their own queues. */
export function createDirtyMaintenance({enabled,run,onError}){
 let pending=null,dirty=false,disposed=false,all=false;const keys=new Set();
 const schedule=()=>pending=Promise.resolve().then(async()=>{
  while(dirty&&!disposed){
   const scope=all?null:new Set(keys);dirty=false;all=false;keys.clear();
   if(enabled())try{await run(scope)}catch(error){onError(error)}
  }
 }).finally(()=>{pending=null;if(dirty&&!disposed&&enabled())return schedule()});
 const request=(key=null)=>{
  if(disposed||!enabled())return Promise.resolve();
  dirty=true;if(key===null){all=true;keys.clear()}else if(!all)keys.add(key);
  return pending??schedule();
 };
 return {request,dispose(){disposed=true;dirty=false;all=false;keys.clear()}};
}

/** Skip only explicitly unrelated fields; imports and unknown changes still run. */
export function isUnrelatedMaintenanceUpdate(changes,ignored){
 const paths=[];
 const visit=(value,path)=>{
  if(value&&typeof value==='object'&&!Array.isArray(value)&&Object.keys(value).length){for(const [key,child]of Object.entries(value))visit(child,path?`${path}.${key}`:key)}
  else if(path)paths.push(path);
 };
 visit(changes,'');
 return paths.length>0&&paths.every(path=>ignored.some(prefix=>path===prefix||path.startsWith(prefix+'.')));
}
export const COSMETIC_UPDATE_FIELDS=Object.freeze(['name','img','sort','_id','_stats.modifiedTime','_stats.lastModifiedBy']);
