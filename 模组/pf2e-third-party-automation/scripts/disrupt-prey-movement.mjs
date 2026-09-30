import {getDisruptPreyReadyMeleeStrikes,isCurrentDisruptToken,isDisruptPreyTarget} from './disrupt-prey-rules.mjs';

/**
 * Add conservative native departure checkpoints before Foundry freezes a path.
 * These boxes only reduce needless segment updates. At each checkpoint the
 * adapter must use the real current Tokens for reach, walls and event checks.
 * Never mutate the native path, fabricate a future Token, or execute movement.
 */
export function prepareDisruptMovementCheckpoints({game,token,path,reactors}){
 if(!isCurrentDisruptToken(token,game))throw Error('移动来源Token已不存在。');
 if(!Array.isArray(path)||path.some(w=>!Number.isFinite(w?.x)||!Number.isFinite(w?.y)||
  w.width!==undefined&&(!Number.isFinite(w.width)||w.width<=0)||w.height!==undefined&&(!Number.isFinite(w.height)||w.height<=0)))throw Error('原生移动路径坐标无效。');
 const grid=token.parent.grid,boxes=[];
 for(const reactor of reactors??[]){
  if(!isCurrentDisruptToken(reactor.token,game)||reactor.token.parent!==token.parent||reactor.token.actor.uuid!==reactor.actor?.uuid||!isDisruptPreyTarget(reactor.actor,token,game))continue;
  const strikes=getDisruptPreyReadyMeleeStrikes(reactor.actor);if(!strikes.length)continue;
  const reach=Math.max(...strikes.map(s=>s.reach)),bounds=reactor.token.object.bounds;
  if(!bounds||![bounds.x,bounds.y,bounds.width,bounds.height].every(Number.isFinite))continue;
  boxes.push({bounds,reach});
 }
 if(!boxes.length)return path.map(w=>({...w}));
 const square=grid?.type===1&&Number.isFinite(grid.size)&&grid.size>0&&Number.isFinite(grid.distance)&&grid.distance>0;
 return path.map(waypoint=>{
  const copy={...waypoint};
  // The campaigns use square grids. For another topology preserve correctness
  // with native per-waypoint checkpoints until a native spatial bound is known.
  const near=!square||boxes.some(({bounds,reach})=>{
   const pad=reach/grid.distance*grid.size,w=(waypoint.width??token.width)*grid.size,h=(waypoint.height??token.height)*grid.size;
   return waypoint.x+w>=bounds.x-pad&&waypoint.x<=bounds.x+bounds.width+pad&&
    waypoint.y+h>=bounds.y-pad&&waypoint.y<=bounds.y+bounds.height+pad;
  });
  if(near)copy.checkpoint=true;
  return copy;
 });
}
