// Two corrections for PF2e Toolbelt's auto-cover (3.60+), applied to its tool instance.
//
// walls: the default wall mode "Cover if Movement" is documented as "the wall will provide cover if it
//   also restricts movement", but the filter tests `edge.isLimited("move")`. Core's Edge#isLimited is
//   `this[type] === LIMITED` and a wall's movement is only ever NONE or NORMAL, so nothing is excluded:
//   open doors and walls that only block sight, light or sound give standard cover. Here "move" is
//   resolved per wall to "normal" (currently restricts movement) or "none" before Toolbelt filters.
//
// stealth: with PF2e Stealthy's `useToolbeltCover`, Stealthy adds each observer's cover to the Hide or
//   Sneak total itself. Toolbelt also pushes a cover modifier into that check when it has a target, so
//   the targeted observer counts the same cover twice (Stealthy's addition is plain arithmetic on the
//   total and never meets PF2e's circumstance stacking). While Stealthy owns it, the cover Toolbelt
//   computes for skill checks is reported as none; attack rolls are unchanged.

const TOOLBELT='pf2e-toolbelt';
const STEALTHY='pf2e-stealthy';
const patched=new WeakSet();

// Toolbelt keeps its tool instances on `game.toolbelt.debug.tools`, readable only while its debug flag is
// on. The flag is restored before returning, so no debug drawing happens.
export function findAutoCoverTool(runtime=globalThis){
  const debug=runtime.CONFIG?.debug;
  if(!debug||typeof debug!=='object')return null;
  const had=Object.hasOwn(debug,TOOLBELT),previous=debug[TOOLBELT];
  try{
    debug[TOOLBELT]=true;
    return runtime.game?.toolbelt?.debug?.tools?.autoCover??null;
  }finally{
    if(had)debug[TOOLBELT]=previous;else delete debug[TOOLBELT];
  }
}

export function wallRestrictsMovement(wall,runtime=globalThis){
  const none=runtime.CONST?.EDGE_SENSE_TYPES?.NONE??0;
  // The document's edge already reflects door state: core sets every sense of an open door to NONE.
  const edge=wall?.edge;
  if(edge&&typeof edge==='object'&&'move' in edge)return edge.move!==none;
  if(wall?.isOpen)return false;
  return wall?.move!==undefined&&wall.move!==none;
}

export function stealthyOwnsCover(runtime=globalThis){
  if(!runtime.game?.modules?.get?.(STEALTHY)?.active)return false;
  try{return runtime.game.settings.get(STEALTHY,'useToolbeltCover')===true;}catch{return false;}
}

export function installToolbeltCoverPatch({runtime=globalThis,report=()=>{}}={}){
  const stats={wallsWithCover:0,wallsWithoutCover:0,stealthDeferred:0};
  const parts={};
  const finish=status=>{const result={feature:'toolbeltCover',status,parts,stats};report(result);return result;};
  if(!runtime.game?.modules?.get?.(TOOLBELT)?.active){
    parts.walls=parts.stealth={status:'skipped',reason:'module-inactive'};
    return finish('skipped');
  }
  const tool=findAutoCoverTool(runtime);
  if(!tool||tool.key!=='autoCover'){
    // Toolbelt is active but its tool can't be reached: both parts should be running, so say so.
    parts.walls=parts.stealth={status:'unsupported-runtime',reason:'tool-not-found'};
    return finish('unsupported-runtime');
  }
  if(patched.has(tool))return finish('already-installed');

  const getWallMode=tool.getWallMode;
  if(typeof getWallMode==='function'){
    tool.getWallMode=function(wall){
      const mode=getWallMode.call(this,wall);
      if(mode!=='move')return mode;
      const blocks=wallRestrictsMovement(wall,runtime);
      stats[blocks?'wallsWithCover':'wallsWithoutCover']++;
      return blocks?'normal':'none';
    };
    parts.walls={status:'installed'};
  }else parts.walls={status:'skipped',reason:'source-changed'};

  const calculateCover=tool.calculateCover;
  if(typeof calculateCover==='function'){
    tool.calculateCover=function(origin,target,options={}){
      const cover=calculateCover.call(this,origin,target,options);
      if(options?.context?.type!=='skill-check'||!stealthyOwnsCover(runtime)||typeof cover?.updateLevel!=='function')return cover;
      if(typeof cover.hasCover==='boolean'&&!cover.hasCover)return cover;
      stats.stealthDeferred++;
      return cover.updateLevel('none');
    };
    parts.stealth={status:'installed'};
  }else parts.stealth={status:'skipped',reason:'source-changed'};

  patched.add(tool);
  const installed=Object.values(parts).some(part=>part.status==='installed');
  return finish(installed?'installed':'skipped');
}
