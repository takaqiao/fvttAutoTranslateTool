import {MODULE_ID} from './rules.mjs';
import {buildRepairPlan} from './repairs.mjs';

export function repairCandidates(game){
 return Array.from(game.actors?.party?.members??[]).filter(actor=>{
  if(actor.type!=='character'||actor.flags?.[MODULE_ID]?.autoRepairDisabled)return false;
  const plan=buildRepairPlan(actor.toObject());
  return plan.updates.length>0||plan.missingSpells.some(spell=>spell.spellcastingEntryId);
 });
}

export function createMaintenance({game,repair}){
 const pending=new Map();
 const run=async actor=>{
  if(game.user!==game.users.activeGM)return;
  const candidates=repairCandidates(game);
  if(!actor)return Promise.all(candidates.map(run));
  if(!candidates.some(candidate=>candidate.uuid===actor.uuid))return;
  if(pending.has(actor.uuid))return pending.get(actor.uuid);
  const work=Promise.resolve().then(()=>repair(actor));pending.set(actor.uuid,work);
  try{return await work}finally{if(pending.get(actor.uuid)===work)pending.delete(actor.uuid)}
 };
 return run;
}
