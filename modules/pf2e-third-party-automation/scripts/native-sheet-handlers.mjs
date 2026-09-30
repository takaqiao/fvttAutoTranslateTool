import {MODULE_ID} from './rules.mjs';

const registries=new WeakMap();
/** A module may register a libWrapper path once. Native sheets return the same
 * action table their listener awaits, so compose observations at that boundary. */
export function registerNativeSheetHandlers(libWrapper,path,factory){
 let registry=registries.get(libWrapper);if(!registry){registry=new Map();registries.set(libWrapper,registry);}
 let entry=registry.get(path);
 if(!entry){
  entry={factories:new Set()};registry.set(path,entry);
  try{libWrapper.register(MODULE_ID,path,function(wrapped,...args){
   let handlers=wrapped(...args);for(const transform of entry.factories)handlers=transform(this,handlers);return handlers;
  },'WRAPPER');}catch(error){registry.delete(path);throw error;}
 }
 entry.factories.add(factory);let released=false;
 return()=>{if(released)return;released=true;entry.factories.delete(factory);if(!entry.factories.size){libWrapper.unregister(MODULE_ID,path);registry.delete(path);}};
}
