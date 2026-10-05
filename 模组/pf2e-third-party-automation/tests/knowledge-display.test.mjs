import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
let api;try{api=await import('../scripts/knowledge-display.mjs');}catch{}
const command=fs.readFileSync(process.env.FVTT_WORKBENCH_RECALL_MACRO??new URL('./fixtures/workbench-7.7.5-recall.txt',import.meta.url),'utf8');
const uuid='Compendium.xdy-pf2e-workbench.asymonous-benefactor-macros-internal.Macro.xcFr7PWwG5OVALNJ';
const colors=['#443730','#171f69','#3c005e','#5e4000','#5e0000'];
const labels={'PF2E.ProficiencyLevel2':'专家（原生）','PF2E.Check.Result.Degree.Check.success':'成功（原生）','PF2E.SkillSociety':'社群（原生）'};
const game={modules:new Map([['xdy-pf2e-workbench',{active:true,version:'7.7.5'}]]),i18n:{lang:'cn',localize:key=>labels[key]??key}},globals={crypto:crypto.webcrypto};
const make=async overrides=>{assert.ok(api?.createWorkbenchDisplay,'scoped Workbench display adapter missing');return api.createWorkbenchDisplay({game,globals,macro:{uuid,type:'script',command},actor:{name:'Actor Skill'},token:{name:'Token Skill'},targets:[{actor:{name:'Creature Skill'}}],...overrides});};
const rank=(value,rank=2)=>`<div class="tag" style="background-color: ${colors[rank]}; white-space:nowrap">${value}</div>`;
test('known producer localizes no-target table text and preserves literal HTML attributes and native skill labels',async()=>{
 const display=await make(),source=`<strong>Recall Knowledge</strong> (Roll: <span style="color: royalblue">12</span>)<table style="font-size: 12px"><tr><th style="padding: 0.25em 0;">Skill</th><th>Prof</th><th>Mod</th><th>Result</th></tr><tr><th>Skill</th><td class="tags">${rank('EXPERT')}</td><td>+13</td><td><span style="color: royalblue">25</span></td></tr></table>`;
 const result=display.content(source);assert.equal(display.enabled,true);assert.match(result,/<strong>回忆知识<\/strong>/);assert.match(result,/<th[^>]*>技能<\/th><th>熟练<\/th><th>调整值<\/th><th>结果<\/th>/);assert.match(result,/<tr><th>Skill<\/th>/);assert.ok(result.includes(rank('专家（原生）')));assert.deepEqual(result.match(/<[^>]*>/g),source.match(/<[^>]*>/g));assert.deepEqual(result.match(/(?:12|\+13|25)/g),source.match(/(?:12|\+13|25)/g));
});
test('target rows localize degrees, ordinal attempts and native rank while keeping names, DC values and tooltip attributes',async()=>{
 const display=await make(),source=`<strong>Recall Knowledge</strong> (Roll: <span>12</span>)<br/><strong>vs. Creature Skill</strong><table><tr><td></td><td></td><td></td><th>1st</th><th>2nd</th><th>3rd</th><th>4th</th></tr><tr><th>Skill</th><td></td><td></td><th>DC 20</th></tr><tr><th>原生社群</th><td>${rank('E')}</td><td>25</td><td><span style="text-decoration:line-through;">Fail</span><br/><span style="color:royalblue" data-tooltip="Custom Skill > Rule">Suc</span></td></tr></table><table><tr><th>Lore Skill DCs</th><th>1st</th><th>6th</th></tr><tr><th>Unspecific</th><td>18</td></tr><tr><th>Specific</th><td>15</td></tr></table>`;
 const result=display.content(source);assert.match(result,/对抗 Creature Skill/);assert.match(result,/第1次/);assert.match(result,/第6次/);assert.match(result,/非专门学识/);assert.match(result,/专门学识/);assert.match(result,/成功（原生）/);assert.match(result,/失败/);assert.ok(result.includes(rank('专家（原生）')));assert.deepEqual(result.match(/<[^>]*>/g),source.match(/<[^>]*>/g));assert.ok(result.includes('data-tooltip="Custom Skill > Rule"'));assert.ok(result.includes('DC 20'));
});
test('scoped notices translate only exact producer sentences and preserve custom names and UUID syntax',async()=>{
 const display=await make(),known='Compendium.pf2e.feats-srd.Item.1Bt7uCW2WI4sM84P',source=`<strong>Recall Knowledge</strong> (Roll: <span>12</span>)<p>Token Skill has @UUID[${known}]</p><p>Other Actor has @UUID[${known}]</p><p>Token Skill has a custom sentence</p>`;
 assert.equal(display.content(source),`<strong>回忆知识</strong>（骰点：<span>12</span>）<p>Token Skill拥有@UUID[${known}]</p><p>Other Actor has @UUID[${known}]</p><p>Token Skill has a custom sentence</p>`);assert.equal(display.notice("Token Skill tries to remember if they've heard something related to this."),'Token Skill尝试回忆相关知识。');assert.equal(display.notice("Other Actor tries to remember if they've heard something related to this."),"Other Actor tries to remember if they've heard something related to this.");assert.equal(display.notice('No selected token or assigned character'),'未选中棋子或指定角色。');
});
test('an unknown Workbench label still translates the audited macro',async()=>{
 const display=await make({game:{...game,modules:new Map([['xdy-pf2e-workbench',{active:true,version:'7.8.0'}]])}});
 assert.equal(display.enabled,true);assert.equal(display.notice('No selected token or assigned character'),'未选中棋子或指定角色。');
});
test('source or module replacement during its digest preserves native output',async()=>{
 for(const change of ['command','module']){
  const module={active:true,version:'future'},localGame={...game,modules:new Map([['xdy-pf2e-workbench',module]])},macro={uuid,type:'script',command};
  const globals={crypto:{subtle:{async digest(...args){const result=await crypto.webcrypto.subtle.digest(...args);if(change==='command')macro.command+='\n';else localGame.modules.set('xdy-pf2e-workbench',{...module});return result}}}};
  const display=await make({game:localGame,macro,globals});assert.equal(display.enabled,false);
 }
});
test('unknown source identity, source fingerprint or changed current text preserves native output',async()=>{
 const source='<strong>Recall Knowledge</strong><table><tr><th>Skill</th></tr></table>';
 for(const overrides of [{game:{...game,i18n:{lang:'en'}}},{game:{...game,modules:new Map([['xdy-pf2e-workbench',{active:false}]])}},{macro:{uuid:'Macro.custom',type:'script',command}},{macro:{uuid,type:'script',command:command+'\n'}}]){const display=await make(overrides);assert.equal(display.enabled,false);assert.equal(display.content(source),source);assert.equal(display.notice('No selected token or assigned character'),'No selected token or assigned character');}
 const display=await make();assert.equal(display.content('<strong>Recall Knowledge (Changed)</strong><table><tr><th>Skill</th></tr></table>'),'<strong>Recall Knowledge (Changed)</strong><table><tr><th>Skill</th></tr></table>');
});
test('exact macro input accepts only CRLF normalization without changing its source',async()=>{
 const display=await make();assert.equal(display.enabled,true);assert.equal(api.WORKBENCH_DISPLAY_SOURCE.sha256,'f61ae3184e46613792d16e7152ba535eaba47c81992e93493558ee8108741eb3');assert.equal(api.knowledgeNativeLabel(game,'society'),'社群（原生）');
});
