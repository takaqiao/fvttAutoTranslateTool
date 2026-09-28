export const GRID_ID='f2e-grid-enhancements';
const HOTFIX_ID='av-v14-hotfix';
const AURA_TARGET='CONFIG.F2e.Aura.token.containsToken';
export const GRID_HASHES={distanceTo:'875b608392a52a31f54418d16f4f3d90d9808cfde0847e89b5c56e2dfa8baab4',onOppositeSides:'e4d352b888a70a5bdc5da63114d48be33169b14e69a2597516b51b480c715c22'};
export const GRID_VERSION_HASHES={
  '8.5.0':GRID_HASHES,
  '8.5.1':{...GRID_HASHES,onOppositeSides:'9be1553623efbd868961251afb354dc40dc9ad0438561811a0a1e2dc5d759160'}
};
// Official 2.2.2, commit 7c26435b. Include dispatch and every grid-specific
// implementation so an unchanged version number cannot authorize changed code.
const GRID_222_SOURCES={
  'main.js':'f16b81c1e6d200763f31ee89b47798548ccba69f03b334fa272b66ff1f20813c',
  'grid/all.js':'db2b53616a6593d4384d64dab82c9c49113cbe69e8c64723d340b4fb3b3e5ac0',
  'grid/square.js':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
  'grid/hex.js':'df6d2995fc8d3638495db0505116a779bf97b736ffd9d7e0e97c4f1743fc0a2a',
  'grid/gridless.js':'2dc7aeb5a53734d45a081b2cf8aed48a51316de8cce4e33ebc6b1d54f8296a8d'
};
// Official 2.3.0 changes only grid/all.js: distances are no longer reduced by
// reach, and aura containment compares the real distance with its radius.
const GRID_SOURCE_PROFILES={
  '2.2.2':GRID_222_SOURCES,
  '2.3.0':{...GRID_222_SOURCES,'grid/all.js':'3f9aab298d3acf75bb2597a0267f858a35b324c5a95e0251dd7501e3d16209ee'}
};

async function verifyGridSources(g,hash,sources){
  if(typeof g.fetch!=='function')return false;
  try{
    const checks=await Promise.all(Object.entries(sources).map(async([path,expected])=>{
      const response=await g.fetch(`modules/${GRID_ID}/scripts/${path}`,{cache:'no-store'});
      return response.ok&&await hash(await response.text())===expected;
    }));
    return checks.every(Boolean);
  }catch{return false;}
}

// Capture before libWrapper.Ready: Grid Enhancements installs its wrappers there.
export function captureGridNative(g=globalThis){
  const proto=g.CONFIG?.Token?.objectClass?.prototype;
  const native=proto ? Object.fromEntries(Object.keys(GRID_HASHES).map(k=>[k,proto[k]])) : {};
  native.conflicts=[];
  native.auraTargets=new WeakSet();native.auraIds=new Set();native.auraConflicts=[];
  g.Hooks?.on('libWrapper.Register',(id,target,_type,_options,targetId)=>{
    if(id!==GRID_ID&&/(?:Token\.object|Token\.objectClass\.prototype)\.(?:distanceTo|onOppositeSides)$/.test(target))native.conflicts.push(id);
    if(id===GRID_ID&&target===AURA_TARGET){
      const prototype=g.CONFIG?.F2e?.Aura?.token;
      if(prototype&&(typeof prototype==='object'||typeof prototype==='function'))native.auraTargets.add(prototype);
      if(Number.isSafeInteger(targetId))native.auraIds.add(targetId);
    }else if(id!==GRID_ID&&id!==HOTFIX_ID&&(target===AURA_TARGET||native.auraIds.has(targetId)))native.auraConflicts.push(id);
  });
  return native;
}

function installGridAuraGuard({g,native,report,unchanged}){
  const installed=new WeakSet();
  const install=()=>{
    const prototype=g.CONFIG?.F2e?.Aura?.token;
    if(!prototype||!native.auraTargets?.has(prototype)||installed.has(prototype))return;
    if(!unchanged())return report('gridAura','source-changed');
    if(native.auraConflicts.length)return report('gridAura','disabled-conflict',native.auraConflicts);
    if(typeof g.libWrapper?.register!=='function')return report('gridAura','unsupported-runtime');
    // Grid creates this alias in a debounced refreshToken callback. Its public
    // registration event fires after its wrapper is added, so our guard can be
    // installed here without racing CONFIG.F2e.Aura creation or changing timing.
    installed.add(prototype);
    try{
      g.libWrapper.register(HOTFIX_ID,AURA_TARGET,function(wrapped,token,...args){
        if(unchanged()&&!native.auraConflicts.length&&g.CONFIG?.F2e?.Aura?.token===prototype
          &&Object.getPrototypeOf(this)===prototype){
          const source=this.token;
          // PF2e's native containsToken rejects undrawn documents before using
          // their objects; Grid omitted that check. Keep valid collision logic.
          if(!source?.hidden&&!token?.hidden&&(!source?.object||!token?.object))return false;
        }
        return wrapped(token,...args);
      },'MIXED');
      report('gridAura','installed');
    }catch(error){installed.delete(prototype);report('gridAura','failed',String(error));}
  };
  g.Hooks.on('libWrapper.Register',(id,target,_type,_options,targetId)=>{
    if(id===GRID_ID&&target===AURA_TARGET)install();
    else if(id!==HOTFIX_ID&&id!==GRID_ID&&(target===AURA_TARGET||native.auraIds.has(targetId)))
      report('gridAura','disabled-conflict',id);
  });
  report('gridAura','waiting-for-grid');
  install();
}

export function adaptGridToken(token,native,g=globalThis,enabled=()=>true,keys=Object.keys(GRID_HASHES)){
  if(keys.some(k=>Object.hasOwn(token,k)))return false;
  for(const key of keys){
    const fallback=token[key];
    Object.defineProperty(token,key,{configurable:true,writable:true,value:function(...args){
      const grid=g.canvas?.grid;
      if(enabled()&&g.canvas?.ready&&grid){
        if(key==='distanceTo'&&grid.isSquare&&!args[1]?.collision_types?.length)
          return native[key].apply(this,args);
        const type=grid.isSquare?'square':grid.isHexagonal?'hex':grid.isGridless?'gridless':null;
        if(key==='onOppositeSides'&&type&&!g.game.settings.get(GRID_ID,`flanking-${type}-override`))
          return native[key].apply(this,args);
      }
      return fallback.apply(this,args);
    }});
  }
  return true;
}

export async function installGrid({g=globalThis,native,hash,report}){
  const module=g.game.modules.get(GRID_ID),system=g.game.system;
  if(!module?.active)return report('grid','inactive');
  const version=system.version,gridVersion=module.version;
  const expectedHashes=Object.hasOwn(GRID_VERSION_HASHES,version)?GRID_VERSION_HASHES[version]:null;
  const sources=Object.hasOwn(GRID_SOURCE_PROFILES,gridVersion)?GRID_SOURCE_PROFILES[gridVersion]:null;
  if((!sources&&!['2.2.0','2.2.1'].includes(gridVersion))||!expectedHashes)return report('grid','unsupported-version');
  // 2.2.2/2.3.0 already repair native flanking delegation. Keep ordinary square
  // distances exactly PF2e-native (2.3.0 still differs beyond reach), and leave
  // their onOppositeSides plus collision-aware distance calculations untouched.
  const keys=sources?['distanceTo']:Object.keys(GRID_HASHES);
  const originals=Object.fromEntries(keys.map(key=>[key,native[key]]));
  const unchanged=()=>g.game.modules.get(GRID_ID)===module&&module.active&&module.version===gridVersion
    &&g.game.system===system&&system.version===version&&keys.every(key=>native[key]===originals[key]);
  if(native.conflicts?.length)return report('grid','disabled-conflict',native.conflicts);
  for(const key of keys)
    if(typeof native[key]!=='function'||await hash(native[key].toString())!==expectedHashes[key])return report('grid','unsupported-source',key);
  if(sources&&!await verifyGridSources(g,hash,sources))return report('grid','unsupported-grid-source');
  if(!unchanged())return report('grid','source-changed-during-validation');
  if(native.conflicts?.length)return report('grid','disabled-conflict',native.conflicts);
  if(sources)installGridAuraGuard({g,native,report,unchanged});
  let enabled=true,count=0;
  const adapt=token=>{if(adaptGridToken(token,native,g,()=>enabled,keys))count++;};
  g.Hooks.on('drawToken',adapt);
  g.Hooks.on('canvasReady',()=>{for(const token of g.canvas.tokens.placeables)adapt(token);report('grid',enabled?'installed':'disabled-conflict',{adaptedTotal:count,current:g.canvas.tokens.placeables.filter(t=>Object.hasOwn(t,'distanceTo')).length});});
  // A later patch on the same methods requires a new compatibility review.
  g.Hooks.on('libWrapper.Register',(id,target)=>{
    if(id!==GRID_ID&&/(?:Token\.object|Token\.objectClass\.prototype)\.(?:distanceTo|onOppositeSides)$/.test(target)){
      enabled=false;report('grid','disabled-conflict',id);
    }
  });
  for(const token of g.canvas?.tokens?.placeables??[])adapt(token);
  report('grid','installed',{adapted:count});
}
