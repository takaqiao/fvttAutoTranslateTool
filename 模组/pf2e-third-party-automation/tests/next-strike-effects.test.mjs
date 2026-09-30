import test from 'node:test';
import assert from 'node:assert/strict';
import {createNextStrikeEffectFrame} from '../scripts/next-strike-effects.mjs';

const TUMBLE='Compendium.patreon-v3.effects.Item.Vh5E1Qgp34sTKfVs',MARKER='self:effect:off-guard-tumble-behind',OFF_GUARD='target:condition:off-guard';
function fixture(){
 const writes=[],actor={uuid:'Actor.hero',items:[],async deleteEmbeddedDocuments(type,ids){writes.push({type,ids});this.items=this.items.filter(item=>!ids.includes(item.id))}};
 const effect={id:'tumble',type:'effect',sourceId:TUMBLE,actor,system:{rules:[{key:'TokenMark',slug:'enemy-tumble-behind',uuid:'Scene.s.Token.foe'}],start:{value:100,initiative:15}},toObject(){return {_id:this.id,type:this.type,_stats:{compendiumSource:this.sourceId},system:structuredClone(this.system)}}};
 actor.items.push(effect);
 const strike={item:{id:'sword',uuid:'Actor.hero.Item.sword',type:'weapon',actor}},target={uuid:'Scene.s.Token.foe'};
 const message={flags:{pf2e:{origin:{uuid:strike.item.uuid,type:'weapon',actor:actor.uuid},context:{type:'attack-roll',outcome:'success',options:[MARKER,OFF_GUARD,'attack-only:ignore'],target:{actor:'Actor.foe',token:target.uuid}}}}};
 const begin=()=>createNextStrikeEffectFrame({actor,strike,target});
 return {actor,effect,strike,target,message,writes,begin};
}
test('a completed Strike consumes the exact one-attack effect and freezes only witnessed off-guard for damage',async()=>{
 const f=fixture(),frame=f.begin();
 assert(frame.capture(f.message));await frame.consume();
 f.message.flags.pf2e.context.options=[];
 assert.deepEqual(f.writes,[{type:'Item',ids:['tumble']}]);assert.deepEqual(f.actor.items,[]);
 assert.deepEqual([...frame.damageOptions()],[OFF_GUARD]);
 assert.deepEqual([...frame.damageOptions([MARKER,'item:trait:magical'])],['item:trait:magical',OFF_GUARD]);
 frame.damageOptions().clear();assert(frame.damageOptions().has(OFF_GUARD),'a caller cannot mutate the frozen fact');
});
for(const outcome of ['failure','criticalFailure','criticalSuccess'])test(`a completed ${outcome} also spends Tumble Behind`,async()=>{
 const f=fixture(),frame=f.begin();f.message.flags.pf2e.context.outcome=outcome;frame.capture(f.message);await frame.consume();assert.equal(f.writes.length,1);
});
for(const invalid of ['no-message','no-roll','wrong-weapon','cancelled','unresolved','foreign-strike','missing-weapon-identity'])test(`no effect is spent for an unconfirmed own Strike (${invalid})`,async()=>{
 const f=fixture();
 if(invalid==='foreign-strike')f.strike.item.actor={uuid:'Actor.somebody-else'};
 if(invalid==='no-roll')f.message.flags.pf2e.context.type='damage-roll';
 if(invalid==='wrong-weapon')f.message.flags.pf2e.origin.uuid='Actor.hero.Item.bow';
 if(invalid==='cancelled')f.message.flags.pf2e.context.outcome='cancelled';
 if(invalid==='unresolved')delete f.message.flags.pf2e.context.outcome;
 if(invalid==='missing-weapon-identity'){delete f.strike.item.uuid;delete f.message.flags.pf2e.origin.uuid;}
 const frame=f.begin();assert.equal(frame.capture(invalid==='no-message'?null:f.message),false);assert.equal(await frame.consume(),false);
 assert.deepEqual(f.writes,[]);assert.deepEqual([...frame.damageOptions()],[]);
});
for(const target of [null,{token:'Scene.s.Token.another'}])test(`the next attack spends the effect without inventing off-guard for a mismatched recipient (${target?.token??'none'})`,async()=>{
 const f=fixture(),frame=f.begin();f.message.flags.pf2e.context.target=target;frame.capture(f.message);await frame.consume();
 assert.equal(f.writes.length,1);assert.deepEqual([...frame.damageOptions()],[]);
});
test('a completed attack with no off-guard condition does not manufacture a damage bonus',async()=>{
 const f=fixture(),frame=f.begin();f.message.flags.pf2e.context.options=[MARKER];frame.capture(f.message);await frame.consume();
 assert.deepEqual([...frame.damageOptions()],[]);assert.equal(f.writes.length,1);
});
for(const kind of ['other-source','expired','condition'])test(`lookalikes and unrelated conditions are never removed (${kind})`,async()=>{
 const f=fixture();if(kind==='other-source')f.effect.sourceId='Compendium.other.effects.Item.Vh5E1Qgp34sTKfVs';if(kind==='expired')f.effect.isExpired=true;if(kind==='condition')f.effect.type='condition';
 const frame=f.begin();frame.capture(f.message);await frame.consume();assert.deepEqual(f.writes,[]);assert.equal(f.actor.items[0],f.effect);
 assert(frame.damageOptions().has(OFF_GUARD),'genuine witnessed off-guard does not depend on this effect');
});
test('a newly gained or refreshed use after the attack is not consumed by the old frame or its delayed card',async()=>{
 const f=fixture(),frame=f.begin();frame.capture(f.message);
 f.effect.system.start.value=101;
 const next={...f.effect,id:'new-use'};f.actor.items.push(next);
 await frame.consume();assert.deepEqual(f.writes,[]);assert.deepEqual(f.actor.items,[f.effect,next]);
 assert.equal(frame.damageOptions([MARKER]).has(MARKER),false);
});
test('a replacement with the same ID is not mistaken for the effect that existed at attack time',async()=>{
 const f=fixture(),frame=f.begin();frame.capture(f.message);f.actor.items=[{...f.effect}];await frame.consume();assert.deepEqual(f.writes,[]);
});
test('an effect acquired after the Strike begins belongs to a future attack',async()=>{
 const f=fixture();f.actor.items=[];const frame=f.begin();f.actor.items.push(f.effect);frame.capture(f.message);await frame.consume();assert.deepEqual(f.writes,[]);
});
test('Patreon miss cleanup can remove the effect before our settlement without losing the captured damage condition',async()=>{
 const f=fixture(),frame=f.begin();frame.capture(f.message);f.actor.items=[];await frame.consume();assert.deepEqual(f.writes,[]);assert(frame.damageOptions().has(OFF_GUARD));
});
test('concurrent settlement attempts delete only once',async()=>{
 const f=fixture(),frame=f.begin();frame.capture(f.message);await Promise.all([frame.consume(),frame.consume()]);assert.equal(f.writes.length,1);
});
test('concurrent external deletion is harmless but a real deletion failure stops the activity',async()=>{
 const f=fixture(),frame=f.begin();frame.capture(f.message);f.actor.deleteEmbeddedDocuments=async()=>{f.actor.items=[];throw Error('already removed')};assert.equal(await frame.consume(),true);
 const g=fixture(),failed=g.begin();failed.capture(g.message);g.actor.deleteEmbeddedDocuments=async()=>{g.writes.push('attempt');throw Error('database unavailable')};
 await assert.rejects(failed.consume(),/database unavailable/);await assert.rejects(failed.consume(),/database unavailable/);assert.deepEqual(g.writes,['attempt']);
});
