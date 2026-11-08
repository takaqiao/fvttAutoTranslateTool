import test from 'node:test';
import assert from 'node:assert/strict';
import {findAutoCoverTool,installToolbeltCoverPatch,wallRestrictsMovement} from '../scripts/patches/toolbelt-cover.mjs';

const NONE=0,NORMAL=20;

class FakeCover{
  constructor(level){this.level=level;}
  get hasCover(){return this.level!=='none';}
  updateLevel(level){this.level=level;return this;}
}

// Mirrors the parts of Toolbelt 3.61.3's AutoCoverTool the patch touches: getWallMode resolves the
// per-wall flag against the world default, calculateCover is what #checkRoll and the highlight call.
function makeTool({defaultMode='move',flags=new Map(),level='standard'}={}){
  return {
    key:'autoCover',
    getWallMode(wall){const flag=flags.get(wall)??'default';return flag==='default'?defaultMode:flag;},
    calculateCover(){return new FakeCover(level);},
  };
}

function makeRuntime({tool=makeTool(),toolbelt=true,stealthy=true,useToolbeltCover=true,debug={}}={}){
  const runtime={CONST:{EDGE_SENSE_TYPES:{NONE,LIMITED:10,NORMAL:20}},CONFIG:{debug}};
  runtime.game={
    modules:{get:id=>({'pf2e-toolbelt':{active:toolbelt},'pf2e-stealthy':{active:stealthy}})[id]},
    settings:{get:(ns,key)=>{if(ns==='pf2e-stealthy'&&key==='useToolbeltCover')return useToolbeltCover;throw new Error('not registered');}},
    toolbelt:{get debug(){return runtime.CONFIG.debug['pf2e-toolbelt']===true?{tools:{autoCover:tool}}:undefined;}},
  };
  return runtime;
}

test('the tool is read through Toolbelt debug access and the debug flag is put back',()=>{
  const tool=makeTool();
  const absent=makeRuntime({tool});
  assert.equal(findAutoCoverTool(absent),tool);
  assert.equal(Object.hasOwn(absent.CONFIG.debug,'pf2e-toolbelt'),false);
  const off=makeRuntime({tool,debug:{'pf2e-toolbelt':false}});
  assert.equal(findAutoCoverTool(off),tool);
  assert.equal(off.CONFIG.debug['pf2e-toolbelt'],false);
});

test('"Cover if Movement" walls give cover only while they restrict movement',()=>{
  const tool=makeTool(),runtime=makeRuntime({tool});
  const result=installToolbeltCoverPatch({runtime});
  assert.equal(result.status,'installed');
  assert.equal(result.parts.walls.status,'installed');
  const closed={edge:{move:NORMAL}},open={edge:{move:NONE}},sightOnly={edge:{move:NONE,sight:NORMAL}};
  assert.equal(tool.getWallMode(closed),'normal');
  assert.equal(tool.getWallMode(open),'none');
  assert.equal(tool.getWallMode(sightOnly),'none');
  assert.deepEqual(result.stats,{wallsWithCover:1,wallsWithoutCover:2,stealthDeferred:0});
});

test('explicit per-wall modes and other world defaults pass through unchanged',()=>{
  const flagged={edge:{move:NONE}},flags=new Map([[flagged,'normal']]);
  const tool=makeTool({flags}),runtime=makeRuntime({tool});
  installToolbeltCoverPatch({runtime});
  assert.equal(tool.getWallMode(flagged),'normal');
  const normalWorld=makeTool({defaultMode:'normal'});
  installToolbeltCoverPatch({runtime:makeRuntime({tool:normalWorld})});
  assert.equal(normalWorld.getWallMode({edge:{move:NONE}}),'normal');
});

test('without an initialized edge, door state and the movement field decide',()=>{
  const runtime=makeRuntime();
  assert.equal(wallRestrictsMovement({edge:null,move:NORMAL,isOpen:true},runtime),false);
  assert.equal(wallRestrictsMovement({edge:null,move:NORMAL,isOpen:false},runtime),true);
  assert.equal(wallRestrictsMovement({edge:null,move:NONE,isOpen:false},runtime),false);
});

test('stealth checks leave cover to Stealthy only while Stealthy uses Toolbelt cover',()=>{
  const skill={context:{type:'skill-check'}},attack={context:{type:'attack-roll'}};
  const tool=makeTool(),runtime=makeRuntime({tool});
  const result=installToolbeltCoverPatch({runtime});
  assert.equal(tool.calculateCover({}, {}, skill).level,'none');
  assert.equal(tool.calculateCover({}, {}, attack).level,'standard');
  // The cover highlight calls without a check context.
  assert.equal(tool.calculateCover({}, {}).level,'standard');
  assert.equal(result.stats.stealthDeferred,1);
  for(const options of [{stealthy:false},{useToolbeltCover:false}]){
    const other=makeTool();
    installToolbeltCoverPatch({runtime:makeRuntime({tool:other,...options})});
    assert.equal(other.calculateCover({}, {}, skill).level,'standard');
  }
});

test('reports a quiet skip when Toolbelt is off and a warning state when its tool is unreachable',()=>{
  const inactive=installToolbeltCoverPatch({runtime:makeRuntime({toolbelt:false})});
  assert.equal(inactive.status,'skipped');
  assert.equal(inactive.parts.walls.reason,'module-inactive');
  const missing=makeRuntime();
  missing.game.toolbelt={debug:undefined};
  const result=installToolbeltCoverPatch({runtime:missing});
  assert.equal(result.status,'unsupported-runtime');
  assert.equal(result.parts.walls.status,'unsupported-runtime');
});

test('installs once per tool instance',()=>{
  const tool=makeTool(),runtime=makeRuntime({tool});
  installToolbeltCoverPatch({runtime});
  const wrapped=tool.getWallMode;
  assert.equal(installToolbeltCoverPatch({runtime}).status,'already-installed');
  assert.equal(tool.getWallMode,wrapped);
});
