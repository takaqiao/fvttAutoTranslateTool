import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {buildHuntPreyPatreonRepairs} from '../scripts/knowledge-automation.mjs';

const work='C:/Users/Taka/Desktop/fvtt/work',captured=`${work}/bob2-tpa-gap-research-20261101`;
const patreonRoot=process.env.PATREON_3232_ROOT??`${work}/main-compat-20261123/live/modules/patreon-v3`;
const pinned=(file,sha256)=>{const bytes=readFileSync(file);assert.equal(createHash('sha256').update(bytes).digest('hex'),sha256,file);return bytes.toString('utf8');};
const patreon=pinned(`${patreonRoot}/src/index.js`,'74961f3f5fa2e75b7a84367fbe9be4634c9f861f4c5ffccef58da68ea1b97523');
const config=JSON.parse(pinned(`${patreonRoot}/rules/config.json`,'8e6bfd8f36e34b6849719d7547733168d3c805362b12e6138558eb845ebf4c94'));
const rangedChat=readFileSync(`${process.env.RANGED_COMBAT_CURRENT_ROOT??`${captured}/snapshot/modules/pf2e-ranged-combat`}/scripts/utils/chat.js`,'utf8');
const pf=readFileSync(process.env.PF2E_NATIVE_861_BUNDLE??`${captured}/snapshot/system/pf2e.mjs`,'utf8');
const Predicate=Function('M',`let Wn=${pf.match(/Wn = (class Predicate[\s\S]*?), AutomaticBonusProgression\$1 =/)[1]};return Wn;`)(x=>!!x&&typeof x==='object'&&!Array.isArray(x));
const HUNT='cb4a9xC8VCzVtZ9Y',rules=()=>Object.fromEntries(config.groups.map(group=>[group.uuid,structuredClone(group)]));
const handler=(rules,triggerType)=>rules[HUNT].handlerRules.find(rule=>rule.triggerType===triggerType);
// Patreon's trigger type and the options it builds for an action card posted by item.toMessage().
const triggerType=(()=>{const start=patreon.indexOf('function H(a){'),end=patreon.indexOf('r(H,"triggerType")');assert.ok(start>=0&&end>start);return Function(`${patreon.slice(start,end)};return H;`)();})();
const cardOptions=['target','origin:item:hunt-prey','origin:item:slug:hunt-prey','origin:item:sourceId:Compendium.pf2e.actionspf2e.Item.JYi4MnsdFu618hPm','mType:postInfo'];
function game({ranged=true,chat=2}={}){
 return {modules:new Map([['pf2e-ranged-combat',{active:ranged}],['patreon-v3',{active:true}]]),settings:{get:(module,key)=>{assert.equal(`${module}.${key}`,'pf2e-ranged-combat.postActionToChat');return chat;}}};
}

test('Ranged Combat re-posts Hunt Prey with toMessage at the full chat level, which Patreon reads as postInfo',()=>{
 assert.match(rangedChat,/static async postAction\(action\) \{\s*if \(game\.settings\.get\("pf2e-ranged-combat", "postActionToChat"\) == ChatLevel\.full\) \{\s*CONFIG\.pf2eRangedCombat\.chatHook = false;\s*await action\.toMessage\(\);/);
 assert.equal(triggerType({flags:{pf2e:{origin:{actor:'Actor.a',uuid:'Actor.a.Item.h',type:'action',rollOptions:['origin:item:hunt-prey']}}}}),'postInfo');
 assert.equal(triggerType({flags:{pf2e:{context:{type:'self-effect',item:'h'}}}}),'self-effect');
 assert.deepEqual(handler(config.groups.reduce((all,group)=>({...all,[group.uuid]:group}),{}),'postInfo'),{type:'handler',triggerType:'postInfo',value:'huntPrey',target:'none',predicate:['target','origin:item:hunt-prey']});
});

test('with the full Hunt Prey card, Patreon\'s postInfo huntPrey handler no longer matches it',()=>{
 const original=rules();
 assert.equal(Predicate.test(handler(original,'postInfo').predicate,cardOptions),true,'Patreon 212 would add its own Hunt Prey effects');
 const {rules:repaired,changes}=buildHuntPreyPatreonRepairs(original,{game:game()});
 assert.deepEqual(changes.map(change=>change.path),[`${HUNT}.handlerRules.1.predicate`]);
 assert.equal(Predicate.test(handler(repaired,'postInfo').predicate,cardOptions),false);
 assert.deepEqual(handler(repaired,'self-effect'),handler(original,'self-effect'),'the self-effect handler is left as it is');
 assert.deepEqual({...repaired,[HUNT]:null},{...original,[HUNT]:null},'no other group changes');
 assert.deepEqual(original,rules(),'the input is not mutated');
 assert.deepEqual(buildHuntPreyPatreonRepairs(repaired,{game:game()}).changes,[]);
});

for(const [name,options]of [['Ranged Combat is not active',{ranged:false}],['Ranged Combat posts the simple card',{chat:1}],['Ranged Combat posts nothing',{chat:0}]])test(`Patreon's Hunt Prey handler is kept when ${name}`,()=>{
 const original=rules(),result=buildHuntPreyPatreonRepairs(original,{game:game(options)});
 assert.deepEqual(result.changes,[]);assert.deepEqual(result.rules,original);
});

test('a Hunt Prey group without the postInfo handler (rules 211) is left alone',()=>{
 const original=rules();original[HUNT].handlerRules=original[HUNT].handlerRules.filter(rule=>rule.triggerType!=='postInfo');
 assert.deepEqual(buildHuntPreyPatreonRepairs(original,{game:game()}).changes,[]);
});
