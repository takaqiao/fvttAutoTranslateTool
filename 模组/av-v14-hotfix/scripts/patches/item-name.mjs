export const ITEM_NAME_HASHES=Object.freeze({
  '8.5.0':'faa5efaa7bc596b4bee8e1c7d9219044c456ae9bbe7acfdc78062db5245201d7',
  '8.5.1':'28b290589d58d9de5c3eea7a1c05a7a9b4992bb45e5e62816e8f1e97a63a85da'
});

/** Wrapper around the audited PF2e 8.5.0 / 8.5.1 generateItemName implementations.
 * Only its existing no-op branches are moved before weapon-map allocation.
 * All actual generated names are still calculated by the original function.
 */
export function createItemNameFastPath(original,globals,stats={fast:0,delegated:0}){
  const enumerable=(object,key)=>Object.prototype.propertyIsEnumerable.call(object,key);
  return function generateItemNameFastPath(...args){
    const item=args[0];
    const delegate=()=>{stats.delegated++;return original.apply(this,args);};
    if(typeof item?.isOfType!=='function'||!item.isOfType('armor','shield','weapon'))return delegate();
    const base=item.baseType??'';
    if(!base||item.isSpecific){stats.fast++;return item.name;}
    let exists,key;
    if(item.isOfType('armor','shield')){
      const types=item.isOfType('armor')?globals.CONFIG.PF2E.baseArmorTypes:globals.CONFIG.PF2E.baseShieldTypes;
      exists=base in types;key=types[base];
    }else{
      const weapon=globals.CONFIG.PF2E.baseWeaponTypes,shield=globals.CONFIG.PF2E.baseShieldTypes;
      if(enumerable(shield,base)){exists=true;key=shield[base];}
      else if(enumerable(weapon,base)){exists=true;key=weapon[base];}
      else if(base in Object.prototype)return delegate();
      else exists=false;
    }
    if(!exists||item._source.name!==globals._loc(key??'')){stats.fast++;return item.name;}
    return delegate();
  };
}
