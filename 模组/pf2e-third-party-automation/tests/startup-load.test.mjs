import test from 'node:test';
import assert from 'node:assert/strict';
import * as privacy from '../scripts/salubrious-message-privacy.mjs';
import * as runtime from '../scripts/runtime.mjs';
function game(){return {modules:new Map([['xdy-pf2e-workbench',{active:true,version:'7.7.5'}]]),system:{version:'8.5.1'},release:{generation:14},PF2eWorkbench:{refocus(){}}}}
for(const mode of ['missing','inactive','unsupported','api-missing'])test(`Workbench ${mode} never fetches its bundle`,async()=>{
 const g=game();if(mode==='missing')g.modules.clear();if(mode==='inactive')g.modules.get('xdy-pf2e-workbench').active=false;if(mode==='unsupported')g.modules.get('xdy-pf2e-workbench').version='99';if(mode==='api-missing')delete g.PF2eWorkbench;
 let requests=0;const r=await privacy.loadSalubriousWorkbench({game:g,fetch:async()=>{requests++;throw Error('not expected')}});assert.equal(r.ready,false);assert.equal(requests,0);
});
for(const phase of ['fetch','body'])test(`Workbench ${phase} timeout releases initialization and aborts the request`,async()=>{
 let signal;const r=await privacy.loadSalubriousWorkbench({game:game(),timeoutMs:10,fetch:async(_url,options)=>{signal=options.signal;return phase==='fetch'?new Promise(()=>{}):{ok:true,text:()=>new Promise(()=>{})}}});assert.equal(r.ready,false);assert.equal(r.reason,'workbench-verification-timeout');assert.equal(signal.aborted,true);
});
test('Workbench errors and changed source do not grant compatibility',async()=>{
 const g=game();const failed=await privacy.loadSalubriousWorkbench({game:g,fetch:async()=>({ok:false,status:404})});assert.equal(failed.ready,false);
 const changed=await privacy.loadSalubriousWorkbench({game:g,fetch:async()=>({ok:true,text:async()=>{g.modules.get('xdy-pf2e-workbench').active=false;return 'changed'}})});assert.equal(changed.ready,false);
});
test('provider selection short circuits after the first matching action and preserves receiver',()=>{
 const item={},calls=[],first={resolveAction(x){assert.equal(this,first);assert.equal(x,item);calls.push('first')}},second={resolveAction(){calls.push('second');return 'party:guardian'}},third={resolveAction(){throw Error('unrelated provider evaluated')}};
 assert.equal(runtime.resolveProviderAction([first,second,third],item),'party:guardian');assert.deepEqual(calls,['first','second']);assert.equal(runtime.resolveProviderAction([{},first],item),undefined);
});
