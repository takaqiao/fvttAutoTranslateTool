import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {spawnSync,execFileSync} from 'node:child_process';
import {patchStaticReceiver} from '../tools/native-manual-pool-static-receiver/patch.mjs';
import {patchToolbeltManualPool} from '../tools/toolbelt-manual-pool/patch.mjs';

const moduleRoot=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),tool=path.join(moduleRoot,'tools/manual-shared-maintenance.mjs'),hash=b=>createHash('sha256').update(b).digest('hex');
const p2Path=process.env.PF2E_MANUAL_POOL_BATCH_SOURCE??process.env.FVTT_PF2E_BUNDLE,t0Path=process.env.TOOLBELT_MANUAL_SOURCE;
assert.ok(p2Path&&t0Path,'PF2E_MANUAL_POOL_BATCH_SOURCE (or FVTT_PF2E_BUNDLE) and TOOLBELT_MANUAL_SOURCE are required; fixed authorized originals, no skips');
const p2=fs.readFileSync(p2Path),t0=fs.readFileSync(t0Path),p2SHA='d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157',t0SHA='2946fa27eaf0963098f9f48ee48f777c5cf65166b56b043de9404f09813c240f',p3SHA='9be357c96dff3d0790edcb0d7889db98cfded0f41f34fd161233ea0bdd0dab11',t1SHA='6937743860c8b2fbeef1037bb99792307cb3fbe7a2540ae0b1ac7010a16a5d90';
assert.equal(hash(p2),p2SHA);assert.equal(hash(t0),t0SHA);
const p3=patchStaticReceiver(p2).bytes,t1=patchToolbeltManualPool(t0).bytes;
function fixture(t){
 const parent=fs.mkdtempSync(path.join(os.tmpdir(),'manual-shared-maintenance-')),dataRoot=path.join(parent,'Data'),planDir=path.join(parent,'private-plan'),pf=path.join(dataRoot,'systems/pf2e/pf2e.mjs'),tb=path.join(dataRoot,'modules/pf2e-toolbelt/scripts/main.js');
 fs.mkdirSync(path.dirname(pf),{recursive:true});fs.mkdirSync(path.dirname(tb),{recursive:true});fs.mkdirSync(path.join(dataRoot,'worlds/sentinel'),{recursive:true});
 fs.writeFileSync(pf,p2);fs.writeFileSync(tb,t0);fs.writeFileSync(path.join(dataRoot,'systems/pf2e/system.json'),JSON.stringify({id:'pf2e',version:'8.5.1'}));fs.writeFileSync(path.join(dataRoot,'modules/pf2e-toolbelt/module.json'),JSON.stringify({id:'pf2e-toolbelt',version:'3.56.5'}));fs.writeFileSync(path.join(dataRoot,'worlds/sentinel/actors.db'),'world bytes must never change\r\n');
 t.after(()=>{const absolute=path.resolve(parent);assert.ok(absolute.startsWith(path.resolve(os.tmpdir())+path.sep));assert.equal(path.basename(absolute).startsWith('manual-shared-maintenance-'),true);fs.rmSync(absolute,{recursive:true,force:true});});
 const snapshot=()=>[pf,tb,path.join(dataRoot,'systems/pf2e/system.json'),path.join(dataRoot,'modules/pf2e-toolbelt/module.json'),path.join(dataRoot,'worlds/sentinel/actors.db')].map(file=>({file,sha256:hash(fs.readFileSync(file))}));
 const cli=(action,{assertOffline=true,...extra}={})=>{const args=[tool,action,'--data-root',dataRoot,'--plan-dir',planDir];if(assertOffline)args.push('--operator-assertion','foundry-and-clients-stopped');for(const [key,value]of Object.entries(extra))args.push('--'+key,String(value));return spawnSync(process.execPath,args,{encoding:'utf8',maxBuffer:1e6});};
 const ok=action=>{const result=cli(action);assert.equal(result.status,0,result.stderr);return JSON.parse(result.stdout);};
 const rejected=(action,reason)=>{const result=cli(action);assert.notEqual(result.status,0);assert.match(result.stderr,new RegExp(reason));};
 const manifestFile=()=>path.join(planDir,'plan.json'),tamper=change=>{const data=JSON.parse(fs.readFileSync(manifestFile()));change(data);fs.writeFileSync(manifestFile(),JSON.stringify(data));};
 return {parent,dataRoot,planDir,pf,tb,snapshot,cli,ok,rejected,tamper};
}

test('existing generators produce the actual accepted P3 and T1 bytes',()=>{assert.equal(hash(p3),p3SHA);assert.equal(hash(t1),t1SHA);});
test('Git filtered Toolbelt observer reconstructs the same accepted T1',()=>{
 const exportedRepo=process.env.AUTOMATION_TEST_GIT_REPOSITORY,commit=process.env.AUTOMATION_TEST_GIT_COMMIT??'HEAD';
 const repo=exportedRepo??execFileSync('git',['rev-parse','--show-toplevel'],{cwd:moduleRoot,encoding:'utf8'}).trim();
 const relative=exportedRepo?process.env.AUTOMATION_TEST_GIT_MODULE_PATH+'/tools/toolbelt-manual-pool/observer.js':path.relative(repo,path.join(moduleRoot,'tools/toolbelt-manual-pool/observer.js')).replaceAll('\\','/');
 if(exportedRepo){assert.ok(path.isAbsolute(repo));assert.match(commit,/^[a-f0-9]{40}$/);assert.equal(process.env.AUTOMATION_TEST_GIT_MODULE_PATH,'模组/pf2e-third-party-automation');}
 const raw=fs.readFileSync(path.join(moduleRoot,'tools/toolbelt-manual-pool/observer.js')),filtered=execFileSync('git',['cat-file','--filters',commit+':'+relative],{cwd:repo});
 assert.deepEqual(filtered,raw);assert.equal(hash(Buffer.concat([filtered,t1.subarray(raw.length)])),t1SHA);
});
test('prepare/status/apply twice/restore change only the fixed pair and preserve world bytes',t=>{const f=fixture(t),before=f.snapshot();assert.equal(f.ok('status').pair,'original');const prepared=f.ok('prepare');assert.equal(prepared.protocol,'manual-shared-maintenance.v1');assert.deepEqual(f.snapshot(),before);assert.equal(f.ok('apply').pair,'installed');assert.equal(hash(fs.readFileSync(f.pf)),p3SHA);assert.equal(hash(fs.readFileSync(f.tb)),t1SHA);assert.equal(f.ok('apply').writes,0);assert.deepEqual(f.snapshot().slice(2),before.slice(2));assert.equal(f.ok('restore').pair,'original');assert.deepEqual(f.snapshot(),before);assert.equal(f.ok('restore').writes,0);});
for(const [name,file,version]of [['PF2e','systems/pf2e/system.json','8.5.2'],['Toolbelt','modules/pf2e-toolbelt/module.json','3.57.0']])test(`unknown ${name} version refuses preparation with zero DataRoot writes`,t=>{const f=fixture(t),target=path.join(f.dataRoot,file),data=JSON.parse(fs.readFileSync(target));data.version=version;fs.writeFileSync(target,JSON.stringify(data));const before=f.snapshot();f.rejected('prepare','version-mismatch');assert.deepEqual(f.snapshot(),before);assert.equal(fs.existsSync(f.planDir),false);});
for(const name of ['PF2e','Toolbelt'])test(`unknown ${name} source refuses preparation with zero DataRoot writes`,t=>{const f=fixture(t);fs.appendFileSync(name==='PF2e'?f.pf:f.tb,'\nexternal edit');const before=f.snapshot();f.rejected('prepare','source-mismatch');assert.deepEqual(f.snapshot(),before);assert.equal(fs.existsSync(f.planDir),false);});
test('writes require an explicit operator assertion, without claiming Setup detection',t=>{const f=fixture(t);f.ok('prepare');const before=f.snapshot();for(const action of ['apply','restore']){const result=f.cli(action,{assertOffline:false});assert.notEqual(result.status,0);assert.match(result.stderr,/operator-assertion-required/);assert.deepEqual(f.snapshot(),before);}});
for(const [name,change]of [['target path',plan=>{plan.targets[0].relativePath='worlds/sentinel/actors.db';}],['data root',plan=>{plan.dataRoot=path.dirname(plan.dataRoot);}],['tool identity',plan=>{plan.tools[0].sha256='0'.repeat(64);}]])test(`tampered ${name} rejects the complete pair before its first write`,t=>{const f=fixture(t);f.ok('prepare');f.tamper(change);const before=f.snapshot();f.rejected('apply','plan-mismatch');assert.deepEqual(f.snapshot(),before);});
test('candidate and backup edits reject installation before either target write',t=>{const f=fixture(t);f.ok('prepare');fs.appendFileSync(path.join(f.planDir,'patched-main.js'),'\nexternal edit');const before=f.snapshot();f.rejected('apply','candidate-mismatch');assert.deepEqual(f.snapshot(),before);fs.writeFileSync(path.join(f.planDir,'patched-main.js'),t1);fs.appendFileSync(path.join(f.planDir,'original-pf2e.mjs'),'\nexternal edit');f.rejected('apply','backup-mismatch');assert.deepEqual(f.snapshot(),before);});
test('a late unknown second target prevents any first target apply write',t=>{const f=fixture(t);f.ok('prepare');fs.appendFileSync(f.tb,'\nexternal edit');const before=f.snapshot();f.rejected('apply','source-mismatch');assert.deepEqual(f.snapshot(),before);});
test('restore accepts either partial installation and returns exactly saved originals',t=>{const f=fixture(t);f.ok('prepare');const before=f.snapshot();fs.writeFileSync(f.pf,p3);assert.equal(f.ok('status').pair,'partial');assert.equal(f.ok('restore').writes,1);assert.deepEqual(f.snapshot(),before);fs.writeFileSync(f.tb,t1);assert.equal(f.ok('restore').writes,1);assert.deepEqual(f.snapshot(),before);});
test('a second-file IO failure preserves the partial pair and uncertain evidence for exact restore',async t=>{const f=fixture(t);f.ok('prepare');const before=f.snapshot(),{applyManualShared}=await import('../tools/manual-shared-maintenance.mjs'),write=fs.writeFileSync;
 try{fs.writeFileSync=(file,...args)=>{if(path.resolve(String(file))===f.tb)throw Error('isolated-second-file-IO-failure');return write(file,...args);};assert.throws(()=>applyManualShared({dataRoot:f.dataRoot,planDir:f.planDir,operatorAssertion:'foundry-and-clients-stopped'}),/isolated-second-file-IO-failure/);}finally{fs.writeFileSync=write;}
 assert.equal(hash(fs.readFileSync(f.pf)),p3SHA);assert.equal(hash(fs.readFileSync(f.tb)),t0SHA);assert.deepEqual(f.snapshot().slice(2),before.slice(2));const failed=fs.readdirSync(f.planDir).filter(name=>name.startsWith('failed-'));assert.equal(failed.length,1);const evidence=JSON.parse(fs.readFileSync(path.join(f.planDir,failed[0])));assert.equal(evidence.status,'uncertain');assert.equal(evidence.writes,1);assert.equal(fs.existsSync(path.join(f.planDir,'attempt-'+evidence.attemptId+'.json')),true);assert.equal(f.ok('restore').writes,1);assert.deepEqual(f.snapshot(),before);
});
test('external edits reject restore of both targets and retain the private recovery evidence',t=>{const f=fixture(t);f.ok('prepare');f.ok('apply');fs.appendFileSync(f.tb,'\nexternal edit');const before=f.snapshot();f.rejected('restore','source-mismatch');assert.deepEqual(f.snapshot(),before);assert.equal(hash(fs.readFileSync(path.join(f.planDir,'original-pf2e.mjs'))),p2SHA);assert.equal(hash(fs.readFileSync(path.join(f.planDir,'original-main.js'))),t0SHA);});
test('a plan directory inside DataRoot is rejected without a generated source spill',t=>{const f=fixture(t),inside=path.join(f.dataRoot,'worlds/sentinel/plan'),before=f.snapshot(),result=spawnSync(process.execPath,[tool,'prepare','--data-root',f.dataRoot,'--plan-dir',inside],{encoding:'utf8'});assert.notEqual(result.status,0);assert.match(result.stderr,/private-new-plan-required/);assert.equal(fs.existsSync(inside),false);assert.deepEqual(f.snapshot(),before);});
