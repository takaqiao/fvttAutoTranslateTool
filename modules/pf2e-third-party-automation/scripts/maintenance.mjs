import {MODULE_ID} from './rules.mjs';
import {buildRepairPlan} from './repairs.mjs';

function needsRepair(actor){
  if(actor.type!=='character'||actor.flags?.[MODULE_ID]?.autoRepairDisabled)return false;
  const plan=buildRepairPlan(actor.toObject());
  return plan.updates.length>0||plan.missingSpells.some(spell=>spell.spellcastingEntryId);
}
export function repairCandidates(game){
 return Array.from(game.actors?.party?.members??[]).filter(needsRepair);
}

export function createMaintenance({game,repair}){
 const pending=new Map();
 const run=async actor=>{
  if(game.user!==game.users.activeGM)return;
  if(!actor)return Promise.all(Array.from(game.actors?.party?.members??[]).map(run));
  if(!Array.from(game.actors?.party?.members??[]).includes(actor))return;
  if(pending.has(actor.uuid))return pending.get(actor.uuid);
  const work=Promise.resolve().then(()=>{
   if(game.user!==game.users.activeGM||!Array.from(game.actors?.party?.members??[]).includes(actor)||!needsRepair(actor))return;
   return repair(actor);
  });pending.set(actor.uuid,work);
  try{return await work}finally{if(pending.get(actor.uuid)===work)pending.delete(actor.uuid)}
 };
 return run;
}
