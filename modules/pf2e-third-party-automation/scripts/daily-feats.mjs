import {MODULE_ID} from './rules.mjs';

export const ARCANE_EVOLUTION_SOURCE='Compendium.pf2e.feats-srd.Item.qxh4evekG28Gt1vj';
const DAILIES='pf2e-dailies',KEY=MODULE_ID+'-arcane-evolution',registered=new WeakSet();
const values=c=>Array.from(c?.values?.()??c??[]);
const id=i=>i.id??i._id;
const source=i=>i.sourceId??i._stats?.compendiumSource??i.flags?.core?.sourceId??null;
const signatureBackup=i=>i.flags?.[MODULE_ID]?.arcaneEvolutionSignature;
const temporary=i=>i.flags?.[MODULE_ID]?.arcaneEvolutionTemporary===true;
const ownedFeat=actor=>actor.type==='character'&&values(actor.items).some(i=>i.type==='feat'&&source(i)===ARCANE_EVOLUTION_SOURCE&&!i.isSuppressed&&!i.system?.suppressed);
const entries=actor=>values(actor.items).filter(i=>i.type==='spellcastingEntry'&&i.system.prepared?.value==='spontaneous'&&i.system.tradition?.value==='arcane');
const ranks=entry=>Object.entries(entry?.system.slots??{}).filter(([key,slot])=>/^slot(?:[1-9]|10)$/.test(key)&&slot.max>0).map(([key])=>Number(key.slice(4))).sort((a,b)=>a-b);
function ordinarySpell(item){return item?.type==='spell'&&!item.isCantrip&&!item.isFocusSpell&&!item.isRitual&&!item.system.ritual&&!(item.system.traits?.value??[]).some(t=>['cantrip','focus'].includes(t));}
function repertoire(actor,entryIds){
 return values(actor.items).filter(i=>ordinarySpell(i)&&entryIds.includes(i.system.location?.value)&&!temporary(i));
}
/** A prepared arcane spellbook or an explicit successful-learning record is evidence.
 * A loose item, level-up plan, or the existence of a compendium is not.
 */
export function getArcaneEvolutionLearned(actor){
 const books=new Set(values(actor.items).filter(i=>i.type==='spellcastingEntry'&&i.system.prepared?.value==='prepared'&&i.system.tradition?.value==='arcane'&&!i.flags?.[DAILIES]?.temporary).map(id));
 const result=new Map();
 for(const item of values(actor.items))if(ordinarySpell(item)&&books.has(item.system.location?.value)&&item.system.traits?.traditions?.includes('arcane')&&!item.flags?.[DAILIES]?.temporary){
  const uuid=source(item)??item.uuid;if(uuid)result.set(uuid,{uuid,name:item.name,item});
 }
 for(const record of actor.flags?.[MODULE_ID]?.arcaneEvolutionLearned??[])if(record?.learned===true&&typeof record.uuid==='string'&&/^(?:Compendium\.[^.]+\.[^.]+\.Item\.[^.]+|Actor\.[^.]+\.Item\.[^.]+)$/.test(record.uuid)){
  if(!result.has(record.uuid))result.set(record.uuid,{uuid:record.uuid,name:String(record.name??record.uuid)});
 }
 return [...result.values()];
}
function signatureCandidates(actor,entryIds){return repertoire(actor,entryIds).filter(i=>!i.system.location.signature||signatureBackup(i));}
function restoreSignature(item){
 const backup=signatureBackup(item);if(!backup)return null;
 const update={_id:id(item),[`flags.${MODULE_ID}.-=arcaneEvolutionSignature`]:null};
 // An explicit intervening user edit is not overwritten.
 if(item.system.location?.signature===true){
  if(backup.existed)update['system.location.signature']=backup.before;
  else update['system.location.-=signature']=null;
 }
 return update;
}
function requireFeat(actor){if(!ownedFeat(actor))throw Error('角色没有可用的奥术演化专长。');}

export function createCampaignDailies({fromUuid=globalThis.fromUuid}={}){
 return [{
  key:KEY,label:'奥术演化',
  condition:actor=>actor.type==='character'&&(ownedFeat(actor)||values(actor.items).some(i=>!!signatureBackup(i))),
  rows:actor=>{
   const casting=entries(actor),entryIds=casting.map(id),known=repertoire(actor,entryIds),learned=getArcaneEvolutionLearned(actor).filter(e=>!known.some(i=>source(i)===e.uuid));
   const options=signatureCandidates(actor,entryIds).map(i=>({value:id(i),label:i.name}));
   const rows=[];
   if(casting.length>1)rows.push({type:'select',slug:'entry',label:'本次演化的奥术自发施法栏',save:true,options:casting.map(e=>({value:id(e),label:e.name??id(e)}))});
   rows.push({type:'select',slug:'mode',label:'每日选择',save:true,options:[{value:'signature',label:'法术库中的额外标志性法术'},...(learned.length?[{value:'learned',label:'临时加入已学法术'}]:[])]});
   rows.push({type:'select',slug:'signature',label:'额外标志性法术（仅使用此分支时生效）',save:true,empty:true,options});
   if(learned.length){
    rows.push({type:'select',slug:'learned',label:'已学外部法术（仅使用此分支时生效）',save:true,empty:true,options:learned.map(e=>({value:e.uuid,label:e.name}))});
    rows.push({type:'select',slug:'rank',label:'临时加入的法术环级',save:true,options:[...new Set(casting.flatMap(ranks))].sort((a,b)=>a-b).map(n=>({value:String(n),label:`${n}环`}))});
   }else rows.push({type:'notify',slug:'learned-information',message:'尚无可验证的已学外部奥术法术目录。完成 Learn a Spell 并记录在奥术法术书后，可在此临时加入；不会开放整个合集。'});
   return rows;
  },
  process:async options=>{
   const {actor,rows,updateItem,deleteItem,addItem,messages}=options;requireFeat(actor);
   const casting=entries(actor),entry=casting.length===1?casting[0]:casting.find(e=>id(e)===rows.entry);
   if(!entry)throw Error('请选择明确的奥术自发施法栏。');
   const available=ranks(entry);if(!available.length)throw Error('该施法栏没有可用的非戏法环位。');
   let selected=null,spellData=null;
   if(rows.mode==='signature'){
    selected=signatureCandidates(actor,[id(entry)]).find(i=>id(i)===rows.signature);
    if(!selected)throw Error('请选择此施法栏中尚非永久标志性法术的有效法术。');
    const spellRank=selected.system.location.heightenedLevel??selected.system.level?.value;
    if(!available.includes(spellRank))throw Error('此法术所在环级没有本职法术环位。');
   }else if(rows.mode==='learned'){
    const known=repertoire(actor,casting.map(id));
    const record=getArcaneEvolutionLearned(actor).find(r=>r.uuid===rows.learned);
    if(!record||known.some(i=>source(i)===record.uuid))throw Error('该法术没有已学记录，或已在法术库中，应选择标志性法术分支。');
    const spell=record.item??await fromUuid(record.uuid),rank=Number(rows.rank);
    if(!ordinarySpell(spell)||!spell.system.traits?.traditions?.includes('arcane')||!Number.isInteger(rank)||!available.includes(rank)||rank<(spell.baseRank??spell.system.level?.value??99))throw Error('已学法术的学派、类别或所选环位不合法。');
    spellData=spell.toObject();delete spellData._id;delete spellData.folder;delete spellData.sort;
    spellData.system.location={value:id(entry),heightenedLevel:rank,signature:false};
    spellData.flags={...spellData.flags,[MODULE_ID]:{arcaneEvolutionTemporary:true,learnedSource:record.uuid}};
   }else throw Error('奥术演化每日分支无效。');
   // Every dependent input has been validated before staging any Dailies mutation.
   for(const item of values(actor.items)){
    if(signatureBackup(item)&&id(item)!==id(selected??{})){const update=restoreSignature(item);if(update)updateItem(update);}
    if(temporary(item))deleteItem(item);
   }
   if(selected){
    const before=signatureBackup(selected)??{existed:Object.hasOwn(selected.system.location,'signature'),before:selected.system.location.signature??null};
    updateItem({_id:id(selected),'system.location.signature':true,[`flags.${MODULE_ID}.arcaneEvolutionSignature`]:before});
    messages.add('spells',{label:'奥术演化：额外标志性法术',selected:selected.name});
   }else{
    addItem(spellData,true);messages.add('spells',{label:'奥术演化：临时已学法术',selected:spellData.name});
   }
  },
  rest:({actor,updateItem,removeItem})=>{
   for(const item of values(actor.items)){
    const update=restoreSignature(item);if(update)updateItem(update);
    // Native Dailies removes its own temporary items regardless of feat presence.
    if(temporary(item)&&!item.flags?.[DAILIES]?.temporary)removeItem(id(item));
   }
  },
 }];
}

export function registerCampaignDailies(game){
 if(!game?.modules?.get(DAILIES)?.active)return {status:'inactive'};
 const api=game.dailies?.api;if(typeof api?.registerCustomDailies!=='function')return {status:'unavailable'};
 if(registered.has(api))return {status:'already-registered'};
 const dailies=createCampaignDailies();api.registerCustomDailies(dailies);registered.add(api);return {status:'registered',keys:dailies.map(d=>d.key)};
}
