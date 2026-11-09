import {sourceTokens} from '../source-contract.mjs';

// PF2e 8.7.0 issue #23627: ActiveEffectPF2e.applyChange returns no change for texture.scaleX/scaleY on a
// token without autoscale. That gate is meant for the size-derived scale PF2e queues in the "initial"
// phase; a TokenImage rule element's scale is queued in the "final" phase and is dropped with it, so the
// token keeps its size when such an effect is toggled. PF2e #23641 (merged after 8.7.0) gates only the
// initial phase. This wrapper does the same until a release carries it.

const SCALE_KEYS=new Set(['texture.scaleX','texture.scaleY']);
const TARGET='CONFIG.ActiveEffect.documentClass.applyChange';
const installed=new WeakSet();

// The scale case as 8.7.0 has it, and as #23641 changes it. `targetDoc`, `change` and `options` stand for
// the method's three parameters; names declared inside the case block are compared by position.
const UNFIXED=`{
  if (!targetDoc.autoscale) return {};
  const mirror = Math.sign(Number(foundry.utils.getProperty(targetDoc._source, change.key)));
  change = { ...change, value: mirror * Number(change.value) };
  break;
}`;
const FIXED=UNFIXED.replace('if (!targetDoc.autoscale)','if (change.phase === "initial" && !targetDoc.autoscale)');
const GUARD='if (targetDoc.documentName !== "Token") return super.applyChange(targetDoc, change, options);';
const TAIL='return super.applyChange(targetDoc, change, options);';
const PARAMETERS=['targetDoc','change','options'];

const stringValue=token=>{
  if(token.kind!=='literal'||!/^["'`]/.test(token.value))return null;
  if(token.value[0]==='`'&&token.value.includes('${'))return null;
  return token.value.slice(1,-1);
};

// Token values with string quotes unified, `const` read as `let`, the parameters renamed to their
// position and block-declared names renamed in declaration order. Property names after "." are kept.
function normalize(tokens,parameters){
  const locals=new Map();
  tokens.forEach((token,index)=>{
    if(['let','const','var'].includes(token.value)&&tokens[index+1]?.kind==='identifier'&&!locals.has(tokens[index+1].value))
      locals.set(tokens[index+1].value,`$local${locals.size}`);
  });
  return tokens.map((token,index)=>{
    const text=stringValue(token);
    if(text!==null)return JSON.stringify(text);
    if(token.value==='const')return 'let';
    if(token.kind!=='identifier'||['.','?.'].includes(tokens[index-1]?.value))return token.value;
    if(locals.has(token.value))return locals.get(token.value);
    const position=parameters.indexOf(token.value);
    return position<0?token.value:`$param${position}`;
  });
}
const same=(left,right)=>left.length===right.length&&left.every((value,index)=>value===right[index]);
const expected=source=>normalize(sourceTokens(source),PARAMETERS);

// 'unfixed', 'fixed' or null for anything else, from the source text of a static `applyChange`.
export function scaleBranchShape(source){
  let tokens;
  try{tokens=sourceTokens(source);}catch{return null;}
  if(tokens[0]?.value!=='applyChange'||tokens[1]?.value!=='(')return null;
  const close=tokens[1].match,parameters=tokens.slice(2,close).filter((_,index)=>index%2===0).map(token=>token.value);
  if(parameters.length!==3||tokens.slice(2,close).some((token,index)=>index%2===0?token.kind!=='identifier':token.value!==','))return null;
  const open=close+1;
  if(tokens[open]?.value!=='{'||tokens[open].match!==tokens.length-1)return null;
  const body=tokens.slice(open+1,-1),read=slice=>normalize(slice,parameters);
  // The guard, then one switch over the change key, then the shared tail call.
  const guard=expected(GUARD),tail=expected(TAIL);
  if(!same(read(body.slice(0,guard.length)),guard))return null;
  const switchAt=guard.length;
  if(body[switchAt]?.value!=='switch'||!same(read(body.slice(switchAt+1,switchAt+6)),['(','$param1','.','key',')']))return null;
  const casesOpen=switchAt+6,casesClose=body.findIndex((token,index)=>index>casesOpen&&token.depth===body[casesOpen].depth&&token.value==='}');
  if(body[casesOpen]?.value!=='{'||casesClose<0||!same(read(body.slice(casesClose+1)),tail))return null;
  // Exactly one case group labelled with both scale keys and nothing else, whose body is one block.
  const depth=body[casesOpen].depth+1;
  let block=null,labelled=0;
  for(let index=casesOpen+1;index<casesClose;index++){
    if(body[index].depth!==depth||body[index].value!=='case')continue;
    const labels=[];
    let at=index;
    while(body[at]?.value==='case'&&body[at+2]?.value===':'){labels.push(stringValue(body[at+1]));at+=3;}
    index=at-1;
    if(!labels.some(label=>SCALE_KEYS.has(label)))continue;
    labelled++;
    if(labels.length!==2||!labels.every(label=>SCALE_KEYS.has(label))||body[at]?.value!=='{')return null;
    const end=body.findIndex((token,offset)=>offset>at&&token.depth===body[at].depth&&token.value==='}');
    if(!['case','default','}'].includes(body[end+1]?.value)||end+1>casesClose)return null;
    block=read(body.slice(at,end+1));
  }
  if(labelled!==1||!block)return null;
  if(same(block,expected(UNFIXED)))return 'unfixed';
  if(same(block,expected(FIXED)))return 'fixed';
  return null;
}

export function installTokenImageScalePatch({moduleId='av-v14-hotfix',runtime=globalThis,report=()=>{}}={}){
  const finish=(status,extra)=>{const result={feature:'tokenImageScale',status,...extra};report(result);return result;};
  if(runtime.game?.system?.id!=='pf2e')return finish('unsupported-system');
  const ActiveEffect=runtime.CONFIG?.ActiveEffect?.documentClass,getProperty=runtime.foundry?.utils?.getProperty;
  if(typeof ActiveEffect!=='function'||typeof runtime.libWrapper?.register!=='function'||typeof getProperty!=='function')
    return finish('unsupported-runtime');
  if(installed.has(ActiveEffect))return finish('already-installed');
  // The method must be PF2e's own, unwrapped, with a parent implementation to defer to.
  const own=Object.getOwnPropertyDescriptor(ActiveEffect,'applyChange')?.value;
  if(typeof own!=='function'||typeof Object.getPrototypeOf(ActiveEffect)?.applyChange!=='function')return finish('unsupported-source');
  const shape=scaleBranchShape(Function.prototype.toString.call(own));
  if(shape==='fixed')return finish('skipped',{reason:'upstream-fixed'});
  if(shape!=='unfixed')return finish('unsupported-source');
  // `applied` counts final-phase scale changes passed on for tokens without autoscale.
  const stats={applied:0};
  runtime.libWrapper.register(moduleId,TARGET,function(wrapped,targetDoc,change,...rest){
    if(targetDoc?.documentName!=='Token'||!SCALE_KEYS.has(change?.key)||change.phase==='initial'||targetDoc.autoscale)
      return wrapped(targetDoc,change,...rest);
    // The rest of the native branch: keep the source mirroring, then apply as the parent class does.
    const mirror=Math.sign(Number(getProperty(targetDoc._source,change.key)));
    stats.applied++;
    return Reflect.apply(Object.getPrototypeOf(ActiveEffect).applyChange,this,[targetDoc,{...change,value:mirror*Number(change.value)},...rest]);
  },'MIXED');
  installed.add(ActiveEffect);
  return finish('installed',{stats});
}
