const durationTypes=new Set(['user-declared','item-text','table-convention']);
function record(value){return value!==null&&typeof value==='object'&&[Object.prototype,null].includes(Object.getPrototypeOf(value))}
function field(input,key){const descriptor=Object.getOwnPropertyDescriptor(input,key);if(descriptor&&!Object.hasOwn(descriptor,'value'))throw Error('invalid-manual-activity');return descriptor?.value}
function text(value,max){if(typeof value!=='string'||!value.trim()||value.trim().length>max)throw Error('invalid-manual-activity');return value.trim()}

/** A declaration carries timing constraints, never native execution credentials. */
export function normalizeManualActivity(input){
 if(!record(input))throw Error('invalid-manual-activity');
 const actorUUID=text(field(input,'actorUUID'),200),label=text(field(input,'label'),200),durationSeconds=field(input,'durationSeconds');
 if(!Number.isFinite(durationSeconds)||durationSeconds<0)throw Error('invalid-manual-duration');
 const supplied=field(input,'durationSource');let durationSource={type:'user-declared'};
 if(supplied!==undefined){
  if(!record(supplied)||!durationTypes.has(field(supplied,'type')))throw Error('invalid-manual-duration-source');
  durationSource={type:field(supplied,'type')};const detail=field(supplied,'detail');
  if(detail!==undefined){if(typeof detail!=='string'||detail.trim().length>500)throw Error('invalid-manual-duration-source');durationSource.detail=detail.trim()}
 }
 const result={actorUUID,label,durationSeconds,durationSource,dependsOn:[]};
 const sessionId=field(input,'sessionId');if(sessionId!==undefined)result.sessionId=text(sessionId,200);
 const dependencies=field(input,'dependsOn');
 if(dependencies!==undefined){if(!Array.isArray(dependencies)||[...dependencies].some(id=>typeof id!=='string'||!id.trim()))throw Error('invalid-manual-dependency');result.dependsOn=[...new Set(dependencies.map(id=>id.trim()))]}
 const order=field(input,'order');if(order!==undefined){if(!Number.isSafeInteger(order)||order<0)throw Error('invalid-manual-order');result.order=order}
 for(const key of ['notBefore','observedStart','observedEnd']){const value=field(input,key);if(value!==undefined){if(!Number.isFinite(value))throw Error('invalid-manual-time');result[key]=value}}
 if((result.observedStart===undefined)!==(result.observedEnd===undefined)||result.observedEnd<result.observedStart)throw Error('invalid-manual-interval');
 return result;
}
