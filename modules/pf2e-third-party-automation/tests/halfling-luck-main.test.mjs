import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createNativeChooser,showNativeChoice} from '../scripts/native-context.mjs';

// Exercise the actual bootstrap's provider construction with both real choice
// adapters: a GM-only adapter cannot serve the original player's local Check.
const main=readFileSync(new URL('../scripts/main.mjs',import.meta.url),'utf8');
const setup=main.split(/\r?\n/).find(line=>line.trimStart().startsWith('const halflingLuck='));
assert.ok(setup);
const construct=new Function('game','fromUuid','choose','showNativeChoice','report','createHalflingLuckProvider',setup+'\nreturn halflingLuck;');

test('bootstrap gives the original player a local native Luck choice without requesting GM-only chooser access',async()=>{
 const user={id:'player',active:true,isGM:false},gm={id:'gm',active:true,isGM:true},game={user,users:{activeGM:gm},world:{id:'ujx5r8oipw7ercdr'},system:{version:'8.5.1'}};
 const actor={testUserPermission:()=>true};let dialogs=0;
 const original=globalThis.foundry;
 globalThis.foundry={applications:{api:{DialogV2:{wait:async options=>{dialogs++;return options.buttons[0].callback();}}}}};
 try{
  const provider=construct(game,()=>{},createNativeChooser({game,send:()=>assert.fail('no remote choice needed')}),showNativeChoice,()=>{},args=>args);
  assert.equal(await provider.choose({actor,user,title:'Luck',choices:[{value:'use',label:'Use'},{value:'decline',label:'Decline'}]}),'use');assert.equal(dialogs,1);
 }finally{globalThis.foundry=original}
});

test('bootstrap keeps Luck disabled outside the verified fortress/PF version',()=>{
 for(const game of [{world:{id:'cotct'},system:{version:'8.5.1'}},{world:{id:'ujx5r8oipw7ercdr'},system:{version:'8.5.2'}}])assert.equal(construct(game,()=>{},()=>{},showNativeChoice,()=>{},()=>assert.fail('unsupported provider')),null);
});
