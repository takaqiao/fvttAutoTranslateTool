export const WORKBENCH_DISPLAY_SOURCE=Object.freeze({module:'xdy-pf2e-workbench',version:'7.7.5',uuid:'Compendium.xdy-pf2e-workbench.asymonous-benefactor-macros-internal.Macro.xcFr7PWwG5OVALNJ',sha256:'f61ae3184e46613792d16e7152ba535eaba47c81992e93493558ee8108741eb3'});
const nativeLabel=(game,key,fallback)=>{const value=game.i18n?.localize?.(key);return typeof value==='string'&&/\p{Script=Han}/u.test(value)?value:fallback;};
export const isChineseKnowledgeLocale=game=>/^(cn|zh(?:[-_].*)?)$/i.test(game.i18n?.lang??'');
const skillLabels={arcana:'奥法',crafting:'手艺',medicine:'医疗',nature:'自然',occultism:'神秘',religion:'宗教',society:'社群'};
export function knowledgeNativeLabel(game,statistic,native){if(!isChineseKnowledgeLocale(game))return native??statistic;return typeof native==='string'&&/\p{Script=Han}/u.test(native)?native:nativeLabel(game,`PF2E.Skill${statistic[0]?.toUpperCase()}${statistic.slice(1)}`,skillLabels[statistic]??native??statistic);}
export function knowledgeProficiencyLabel(game){return isChineseKnowledgeLocale(game)?nativeLabel(game,'PF2E.ProficiencyLabel','熟练'):game.i18n?.localize?.('PF2E.ProficiencyLabel')??'Proficiency';}
const rankNames=['UNTRAINED','TRAINED','EXPERT','MASTER','LEGENDARY'],rankColors=['#443730','#171f69','#3c005e','#5e4000','#5e0000'],rankCN=['未受训','受训','专家','大师','传奇'];
const degreeNames=['CrFail','Fail','Suc','CrSuc'],degreeKeys=['criticalFailure','failure','success','criticalSuccess'],degreeCN=['大失败','失败','成功','大成功'];
const headings=new Map([['Skill','技能'],['Prof','熟练'],['Mod','调整值'],['Result','结果'],['Potential Modifiers','潜在调整值'],['Lore Skill DCs','学识技能难度'],['Unspecific','非专门学识'],['Specific','专门学识'],['Lore Skill','学识技能']]);
const ordinals=['1st','2nd','3rd','4th','5th','6th'];
const featIds=['1Bt7uCW2WI4sM84P','XvX1EyxWbbBF32NV'];
const passthrough={enabled:false,content:value=>value,notice:value=>value};
// Retain every tag byte, including quote choice, whitespace, tooltip values and
// source's unmatched </a>. Only verified text-node slots below are editable.
function* fragments(html){
 let start=0;while(start<html.length){const open=html.indexOf('<',start);if(open<0){yield{raw:html.slice(start),tag:false};return;}if(open>start)yield{raw:html.slice(start,open),tag:false};let quote=null,end=open+1;for(;end<html.length;end++){const char=html[end];if(quote){if(char===quote)quote=null;}else if((char==='"'||char==="'")&&html.slice(open+1,end).trimEnd().endsWith('='))quote=char;else if(char==='>')break;}if(end===html.length){yield{raw:html.slice(open),tag:false};return;}yield{raw:html.slice(open,end+1),tag:true};start=end+1;}
}
const attr=(tag,name)=>tag.match(new RegExp(`\\b${name}\\s*=\\s*(["'])(.*?)\\1`,'i'))?.[2];
/** A display-only adapter for this exact installed macro. Unknown producers,
 * unavailable digest APIs and changed headings retain the native output. */
export async function createWorkbenchDisplay({game,macro,actor,token,targets=[],globals=globalThis}){
 const source=WORKBENCH_DISPLAY_SOURCE,module=game.modules.get(source.module);
 const crypto=globals.crypto??globalThis.crypto;if(!isChineseKnowledgeLocale(game)||!module?.active||module.version!==source.version||macro?.uuid!==source.uuid||macro.type!=='script'||typeof macro.command!=='string'||!crypto?.subtle)return passthrough;
 const command=macro.command.replace(/\r\n/g,'\n'),digest=[...new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(command)))].map(byte=>byte.toString(16).padStart(2,'0')).join('');if(digest!==source.sha256)return passthrough;
 const name=token?.name??actor.name,targetNames=new Set(targets.map(target=>target.actor?.name)),featNotices=new Map(featIds.map(id=>[`${name} has @UUID[Compendium.pf2e.feats-srd.Item.${id}]`,`${name}拥有@UUID[Compendium.pf2e.feats-srd.Item.${id}]`]));
 const content=html=>{
  if(typeof html!=='string'||!html.startsWith('<strong>Recall Knowledge</strong> (Roll: <span'))return html;
  const stack=[];let output='',heading=false;
  for(const fragment of fragments(html)){
   if(fragment.tag){const tag=fragment.raw.match(/^<\s*(\/?)\s*([a-z][\w-]*)/i);if(tag){const closing=!!tag[1],name=tag[2].toLowerCase();if(closing){const index=stack.findLastIndex(node=>node.name===name);if(index>=0)stack.splice(index);}else if(!['br','hr','input','img','meta','link'].includes(name)&&!fragment.raw.endsWith('/>')){const table=stack.findLast(node=>node.name==='table');if(name==='tr'&&table)table.row++;stack.push({name,raw:fragment.raw,...(name==='table'?{row:-1,role:null}:{})});}}output+=fragment.raw;continue;}
   const raw=fragment.raw,text=raw.trim(),parent=stack.at(-1),table=stack.findLast(node=>node.name==='table');let translated=text;
   if(parent?.name==='strong'&&!heading&&text==='Recall Knowledge'){translated='回忆知识';heading=true;}
   else if(!stack.length&&heading&&raw===' (Roll: '){output+='（骰点：';continue;}
   else if(!stack.length&&heading&&raw===')'){output+='）';continue;}
   else if(parent?.name==='strong'&&targetNames.has(text.slice(4))&&text.startsWith('vs. '))translated=`对抗 ${text.slice(4)}`;
   else if(parent?.name==='th'&&table){
    if(table.row===0){if(text==='Skill'||text==='Lore Skill')table.role='list';else if(text==='Potential Modifiers')table.role='potential';else if(text==='Lore Skill DCs')table.role='lore-dc';else if(!table.role&&text==='1st')table.role='primary';if(text==='Prof')translated=knowledgeProficiencyLabel(game);else if(headings.has(text))translated=headings.get(text);else if(ordinals.includes(text))translated=`第${ordinals.indexOf(text)+1}次`;}
    else if(table.role==='primary'&&table.row===1&&text==='Skill')translated='技能';
    else if(table.role==='lore-dc'&&['Unspecific','Specific'].includes(text))translated=headings.get(text);
   }else if(parent?.name==='div'&&table&&attr(parent.raw,'class')==='tag'){
    const color=attr(parent.raw,'style')?.match(/background-color:\s*(#[a-f0-9]{6})/i)?.[1],rank=rankColors.indexOf(color);
    if(rank>=0&&(text===rankNames[rank]||text===rankNames[rank][0]))translated=nativeLabel(game,`PF2E.ProficiencyLevel${rank}`,rankCN[rank]);
   }else if(parent?.name==='span'&&table?.role==='primary'&&table.row>=2&&degreeNames.includes(text))translated=nativeLabel(game,`PF2E.Check.Result.Degree.Check.${degreeKeys[degreeNames.indexOf(text)]}`,degreeCN[degreeNames.indexOf(text)]);
   else if(parent?.name==='p'&&featNotices.has(text))translated=featNotices.get(text);
   output+=translated===text?raw:raw.slice(0,raw.indexOf(text))+translated+raw.slice(raw.indexOf(text)+text.length);
  }
  return output;
 };
 const notice=value=>value==='No selected token or assigned character'?'未选中棋子或指定角色。':value===`${name} tries to remember if they've heard something related to this.`?`${name}尝试回忆相关知识。`:value;
 return {enabled:true,content,notice};
}
