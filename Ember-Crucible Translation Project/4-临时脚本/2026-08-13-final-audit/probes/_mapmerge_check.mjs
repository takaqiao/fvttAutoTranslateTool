import { BABELE_DEFAULTS, CRUCIBLE_LAYER, EMBER_LAYER } from '../../../3-常用脚本/extract/mappings.mjs';
function deepMerge(base, ov){ // mimic foundry mergeObject recursive
  const out = structuredClone(base);
  for (const [k,v] of Object.entries(ov)){
    if (v && typeof v==='object' && !Array.isArray(v) && out[k] && typeof out[k]==='object' && !Array.isArray(out[k])) out[k]=deepMerge(out[k],v);
    else out[k]=structuredClone(v);
  }
  return out;
}
for (const [name,layer] of [['CRUCIBLE',CRUCIBLE_LAYER],['EMBER',EMBER_LAYER]]){
  for (const [dt,def] of Object.entries(layer)){
    const base = BABELE_DEFAULTS[dt];
    if (!base) continue;
    const shallow = {...base, ...def};
    const deep = deepMerge(base, def);
    const a=JSON.stringify(shallow), b=JSON.stringify(deep);
    if (a!==b){ console.log('DIFF', name, dt); console.log('  shallow:',a); console.log('  deep   :',b); }
  }
}
console.log('done');
