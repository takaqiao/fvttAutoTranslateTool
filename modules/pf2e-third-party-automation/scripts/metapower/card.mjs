import {MODULE_ID} from './lifecycle.mjs';
const conditions=['Compendium.battlezoo-eldamon-pf2e.conditions.Bi2aHykg6CZrQCnR','Compendium.battlezoo-eldamon-pf2e.conditions.1fZbuJEbVmE3J4XL'];
/** Pure native-link plan. Ordinals are confined to individually reviewed source
 * profiles; unknown powers never enter this renderer. Original descriptive text
 * remains readable, while the selected branch has the only live damage links. */
export function cardLinkPlan(snapshot,links){
 let damageIndex=0;
 const dice={ 'electric-surge':['1','d4','d8'],'anvil-crawler-lightning':['1','d4','d8'],'static-shock':['2','d6','d12'],'electric-shot':['2','d4','d8'] }[snapshot.powerId];
 return links.map(link=>{
  const result={...link};
  if(link.kind==='damage'){
   const n=damageIndex++;
   if(snapshot.powerId==='high-voltage')result.disabled=true;
   if(dice&&n<2){result.disabled=n!==(snapshot.discharge?1:0);result.formula=`(${dice[0]}+${snapshot.level})${dice[snapshot.discharge?2:1]}[electricity]`;}
   if(snapshot.powerId==='static-shock'&&n===2)result.formula=`(2+${snapshot.level})[electricity]`;
   if(snapshot.powerId==='electric-shot'&&n===2)result.formula=`${snapshot.level}[electricity]`;
   if(snapshot.powerId==='reactive-chain'){
    if(!Number.isFinite(snapshot.triggerDamage)||snapshot.triggerDamage<=0)throw Error('Reactive Chain has no confirmed damage basis.');
    result.formula=`${Math.floor(snapshot.triggerDamage/2)}[electricity]`;
   }
  }
  if(link.kind==='area'&&snapshot.area?.type===link.type)result.distance=snapshot.area.distance;
  if(link.kind==='effect'&&snapshot.siphon?.applies&&conditions.some(uuid=>link.uuid===uuid||link.uuid===uuid.replace('.conditions.','.conditions.Item.')))result.disabled=true;
  return result;
 });
}

/** The native Region click handler consumes data-distance and retains its own
 * shape builder, origin, message ID, preview and canvas.regions.placeRegion. */
export function renderMetapowerCard(message,html,{receipt,onClear,onError=console.error}={}){
 const root=html?.[0]??html;if(!root?.querySelectorAll||!receipt)return;
 root.querySelector('.metapower-controls')?.remove();
 const block=document.createElement('div');block.className='metapower-controls';block.setAttribute('role','status');
 const snapshot=receipt.snapshot;
 block.textContent=receipt.kind?`${receipt.kind==='widen'?'增广元素':'虹吸元素'}：仅限紧接的下一次引导威能。`:snapshot?`${snapshot.kind==='widen'?'增广元素':snapshot.kind==='siphoning'?'虹吸元素':'原生威能'} · ${snapshot.discharge?'放电（Charged −1）':'普通分支'}${snapshot.area?` · ${snapshot.area.distance} 尺`:''}${snapshot.siphon?.applies?' · 无类型；逐目标半伤，关联生物特征匹配时全伤；附加效果不生效':''}`:'已记录实际使用。';
 if(receipt.kind&&onClear){const button=document.createElement('button');button.type='button';button.textContent='已采取其他动作／清除待用威能';button.addEventListener('click',event=>{event.preventDefault();event.stopPropagation();Promise.resolve(onClear(receipt)).catch(onError)});block.append(button);}
 (root.querySelector('.message-content')??root).append(block);
 if(!snapshot)return;
 if(root.dataset.metapowerRendered===receipt.nonce)return;
 root.dataset.metapowerRendered=receipt.nonce;
 const anchors=[...root.querySelectorAll('a.inline-roll[data-damage-roll], a.effect-area, a.content-link[data-uuid]')];
 const links=anchors.map(a=>'damageRoll'in a.dataset?{kind:'damage',baseFormula:a.dataset.baseFormula}:a.classList.contains('effect-area')?{kind:'area',type:a.dataset.type,distance:Number(a.dataset.distance)}:{kind:'effect',uuid:a.dataset.uuid});
 const plan=cardLinkPlan(snapshot,links);
 for(let i=0;i<anchors.length;i++){
  const a=anchors[i],p=plan[i];
  if(p.disabled){const span=document.createElement('span');span.textContent=a.textContent;span.title='本次已选分支／虹吸规则使此链接不可用';span.className='metapower-disabled';a.replaceWith(span);continue;}
  if(p.kind==='damage'){
   if(p.formula){a.dataset.baseFormula=p.formula;a.dataset.formula=p.formula;}
   a.dataset.rollOptions=[...new Set([...(a.dataset.rollOptions??'').split(',').filter(Boolean),`${MODULE_ID}:metapower:${message.id}:${receipt.nonce}`])].join(',');
  }
  if(p.kind==='area'&&p.distance!==undefined){a.dataset.distance=String(p.distance);a.setAttribute('title',`${p.distance} ft`);a.textContent=`${p.distance} ft ${p.type}`;}
 }
 for(const check of root.querySelectorAll('[data-pf2-check]'))check.dataset.rollOptions=[...new Set([...(check.dataset.rollOptions??'').split(',').filter(Boolean),`${MODULE_ID}:metapower:${message.id}:${receipt.nonce}`])].join(',');
}
