import {hashSource} from '../source-hash.mjs';

const SOURCE='face593491e47b86dbb6b9df2647b5774abbc63859596f9dbd30467b0c544dbd';
const functionToString=Function.prototype.toString;
const nativeFunction=fn=>typeof fn==='function'&&/^function(?:\s+(?:(?:get|set)\s+)?[\w$]+)?\s*\([^)]*\)\s*\{\s*\[native code\]\s*\}$/.test(Reflect.apply(functionToString,fn,[]));
function formatMethod(Constructor){
  for(let p=Constructor.prototype;p;p=Object.getPrototypeOf(p)){
    const descriptor=Object.getOwnPropertyDescriptor(p,'format');
    if(descriptor)return descriptor.value;
  }
}
const plain=value=>value!==null&&typeof value==='object'&&[null,Object.prototype].includes(Object.getPrototypeOf(value));
function eligibleOptions(options){
  if(!plain(options)||Reflect.ownKeys(options).length!==2||'formatter' in options)return false;
  const style=Object.getOwnPropertyDescriptor(options,'style'),terms=Object.getOwnPropertyDescriptor(options,'maxTerms');
  return style?.enumerable&&terms?.enumerable&&style.value==='narrow'&&terms.value===2;
}
function eligibleComponents(components){
  if(!plain(components))return false;
  for(const key of Reflect.ownKeys(components)){
    const d=Object.getOwnPropertyDescriptor(components,key);
    if(typeof key!=='string'||!d||!('value' in d)||!(typeof d.value==='boolean'||(typeof d.value==='number'&&Number.isFinite(d.value))))return false;
  }
  return true;
}
export function createDurationFastPath(original,runtime=globalThis){
  const Constructor=runtime.Intl.DurationFormat,format=formatMethod(Constructor);
  const nativeFormat=nativeFunction(format);
  let cached,locale;
  const currentLocale=()=>Object.getOwnPropertyDescriptor(runtime.game?.i18n??{},'lang')?.value;
  const supported=()=>runtime.Intl.DurationFormat===Constructor&&nativeFormat&&formatMethod(Constructor)===format;
  return function(...args){
    if(!supported()||!eligibleOptions(args[2])||!eligibleComponents(args[1]))return Reflect.apply(original,this,args);
    const lang=currentLocale();
    if(typeof lang!=='string')return Reflect.apply(original,this,args);
    if(locale!==lang){cached=undefined;locale=lang;}
    if(cached){
      const forwarded=[...args];forwarded[2]={maxTerms:2,style:'narrow',formatter:cached};
      return Reflect.apply(original,this,forwarded);
    }
    // Keep the native component checks and constructor errors in their original order.
    // Warming after the first successful call costs one extra construction per locale.
    const result=Reflect.apply(original,this,args);
    if(typeof result==='string'&&result!=='∞'&&supported()&&currentLocale()===lang){
      try{cached=new Constructor(lang,{style:'narrow'});}catch{/* Optional warm-up never changes the completed call. */}
    }
    return result;
  };
}

export async function installDurationPatch({runtime=globalThis,hash=hashSource,report=()=>{}}={}){
  const finish=(status,extra={})=>{const result={feature:'duration',status,...extra};report(result);return result;};
  const target=runtime.foundry?.data?.CalendarData,descriptor=target&&Object.getOwnPropertyDescriptor(target,'formatDuration');
  const original=descriptor?.value,Constructor=runtime.Intl?.DurationFormat;
  const format=typeof Constructor==='function'&&formatMethod(Constructor);
  if(typeof original!=='function'||!descriptor.writable||!nativeFunction(Constructor)||!nativeFunction(format))return finish('unsupported-runtime');
  if(await hash(Function.prototype.toString.call(original))!==SOURCE)return finish('unsupported-source');
  const current=Object.getOwnPropertyDescriptor(target,'formatDuration');
  if(current?.value!==original||current.writable!==descriptor.writable||current.configurable!==descriptor.configurable||current.enumerable!==descriptor.enumerable||runtime.Intl.DurationFormat!==Constructor||formatMethod(Constructor)!==format)return finish('source-changed-during-validation');
  const replacement=createDurationFastPath(original,runtime);
  Object.defineProperty(target,'formatDuration',{...descriptor,value:replacement});
  return finish('installed',{restore(){
    const live=Object.getOwnPropertyDescriptor(target,'formatDuration');
    if(live?.value===replacement&&live.writable===descriptor.writable&&live.configurable===descriptor.configurable&&live.enumerable===descriptor.enumerable)Object.defineProperty(target,'formatDuration',descriptor);
  }});
}
