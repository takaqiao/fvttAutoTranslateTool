import {clone,emptyLedger,createActivity,validateSession,validateClock} from './schema.mjs';
const transitions={
  planned:['started','blocked','cancelled'],started:['completing','awaiting-evidence','blocked','uncertain','cancelled'],
  completing:['awaiting-evidence','confirmed','blocked','uncertain'],
  'awaiting-evidence':['confirmed','uncertain'],confirmed:[],blocked:[],uncertain:[],cancelled:[]
};
export function createLedger({read,write,isAuthority}) {
  let tail=Promise.resolve();
  const check=()=>{if(!isAuthority())throw Error('active-gm-required')};
  function mutate(fn) {
    const result=tail.then(async()=>{check();const s=clone(await read()??emptyLedger());check();const value=fn(s);check();await write(s);check();return clone(value)});
    tail=result.catch(()=>{});return result;
  }
  const get=async(collection,key)=>clone((await read())[collection]?.[key]??null);
  function transition(collection,key,{expected,patch}) {
    return mutate(s=>{
      const old=s[collection][key];if(!old||!expected.includes(old.state))throw Error('state-conflict');
      const legal=collection==='clocks'?{started:['confirmed','uncertain'],confirmed:[],uncertain:[]}:transitions;
      if(patch.state&&!legal[old.state]?.includes(patch.state))throw Error('illegal-transition');
      const immutable=collection==='clocks'?['id','sessionId','from','to','gmId']:['id','sessionId','providerId','actorUUID','patientUUIDs','hpPoolUUIDs','startedAt','endsAt','source','groupId'];
      if(immutable.some(k=>k in patch&&JSON.stringify(patch[k])!==JSON.stringify(old[k])))throw Error('immutable-provenance');
      s[collection][key]={...old,...clone(patch)};return s[collection][key];
    });
  }
  return {
    createSession:input=>mutate(s=>{const v=validateSession(input);if(s.sessions[v.id])throw Error('duplicate-session');s.sessions[v.id]=v;return v}),
    getSession:key=>get('sessions',key),getActivity:key=>get('activities',key),getClockCommit:key=>get('clocks',key),
    updateSession:(key,patch)=>mutate(s=>{const v=s.sessions[key];if(!v)throw Error('missing-session');if(['id','startedAt','activityIds'].some(k=>k in patch))throw Error('immutable-session');Object.assign(v,clone(patch));return v}),
    insertActivity:input=>mutate(s=>{const a=createActivity(input);if(s.activities[a.id])throw Error('duplicate-activity');if(!s.sessions[a.sessionId])throw Error('missing-session');s.activities[a.id]=a;s.sessions[a.sessionId].activityIds.push(a.id);return a}),
    transitionActivity:(key,options)=>transition('activities',key,options),
    upsertClockCommit:input=>mutate(s=>{const c=validateClock(input);if(s.clocks[c.id])throw Error('duplicate-clock');if(!s.sessions[c.sessionId])throw Error('missing-session');s.clocks[c.id]=c;return c}),
    transitionClockCommit:(key,options)=>transition('clocks',key,options),
    all:async()=>clone(await read()),
    snapshot:async key=>{const s=await read();return clone({session:s.sessions[key]??null,activities:Object.values(s.activities).filter(a=>a.sessionId===key),clocks:Object.values(s.clocks).filter(c=>c.sessionId===key)})}
  };
}
