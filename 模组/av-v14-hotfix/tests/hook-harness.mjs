import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
const fixture = JSON.parse(await readFile(new URL('./fixtures/sundry-core-hooks.json',import.meta.url),'utf8'));

export function nativeHooks() {
  return vm.runInNewContext(`${fixture.findSplice};
    Object.defineProperty(Array.prototype,'findSplice',{value:findSplice});
    ${fixture.Hooks}; Hooks$1`, {CONFIG:{debug:{hooks:false}},CONST:{vtt:'Foundry'},console});
}
