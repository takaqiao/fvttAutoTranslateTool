// 复刻 patchActorUpdateDocuments 的重试逻辑，验证三件事：
//  ① 正常导入：items/effects 必须完整送达（这是本次修复的核心）
//  ② 某个 actor 畸形：整场导入不中断，且只有它被降级
//  ③ 非 import 场景 / 非已知错误：原样抛出，不吞
const MODULE_ID='ember_cn_unofficial';
const isKnownUpdateDiffError=(e)=>String(e?.message??'').includes('getFailure')
  ||String(e?.message??'').includes('One of original or other are not Objects');
const degradeActorUpdatePayload=(u)=>{const d=structuredClone(u);delete d.items;delete d.effects;return d;};

function makeWrapped(original,importMode){
  return async function(updates,...args){
    const sanitized=updates;
    try{ return await original.call(this,sanitized,...args); }
    catch(error){
      if(!isKnownUpdateDiffError(error)||!importMode) throw error;
      const results=[];
      for(const update of sanitized){
        try{ const p=await original.call(this,[update],...args); if(Array.isArray(p))results.push(...p); continue; }
        catch(se){ if(!isKnownUpdateDiffError(se)) throw se; }
        try{ const p=await original.call(this,[degradeActorUpdatePayload(update)],...args);
             if(Array.isArray(p))results.push(...p);
             results.degraded=(results.degraded||0)+1; continue; }
        catch(de){ if(!isKnownUpdateDiffError(de)) throw de; }
      }
      return results;
    }
  };
}
let pass=0,fail=0;
const ck=(n,g,w)=>{ if(JSON.stringify(g)===JSON.stringify(w))pass++; else{fail++;console.log(`  ✗ ${n}: got ${JSON.stringify(g)} want ${JSON.stringify(w)}`);} };

// ① 正常路径：original 全部成功，检查它收到的载荷有没有 items/effects
let seen=null;
const okOriginal=async function(payload){ seen=payload; return payload.map(p=>({_id:p._id})); };
const w1=makeWrapped(okOriginal,true);
await w1.call(null,[{_id:'a1',name:'怪A',items:[{_id:'i1'}],effects:[{_id:'e1'}]}]);
ck('正常导入保留 items', !!seen[0].items, true);
ck('正常导入保留 effects', !!seen[0].effects, true);

// ② 批量失败 → 逐个重试；其中 bad 这个 actor 完整载荷失败、降级后成功
const calls=[];
const flakyOriginal=async function(payload){
  calls.push(structuredClone(payload));
  if(payload.length>1) throw new Error('getFailure batch');
  const u=payload[0];
  if(u._id==='bad'&&u.items) throw new Error('One of original or other are not Objects');
  return [{_id:u._id}];
};
const w2=makeWrapped(flakyOriginal,true);
const r2=await w2.call(null,[
  {_id:'good',items:[{_id:'i1'}],effects:[]},
  {_id:'bad', items:[{_id:'i2'}],effects:[]},
]);
ck('两个 actor 都导入成功', r2.map(x=>x._id), ['good','bad']);
const goodCall=calls.find(c=>c.length===1&&c[0]._id==='good');
ck('good 保住了 items', !!goodCall[0].items, true);
const badFinal=calls.filter(c=>c.length===1&&c[0]._id==='bad').pop();
ck('bad 才被降级（无 items）', badFinal[0].items===undefined, true);
ck('只降级了 1 个', r2.degraded, 1);

// ③ 非导入场景下的已知错误必须原样抛
const w3=makeWrapped(async()=>{throw new Error('getFailure');},false);
let threw=false; try{ await w3.call(null,[{_id:'x'}]); }catch{ threw=true; }
ck('非导入场景不吞错', threw, true);
// ④ 未知错误即使在导入场景也必须抛
const w4=makeWrapped(async()=>{throw new Error('某个真 bug');},true);
threw=false; try{ await w4.call(null,[{_id:'x'}]); }catch{ threw=true; }
ck('未知错误不吞', threw, true);

console.log(`\n通过 ${pass} / 失败 ${fail}`);
process.exit(fail?1:0);
