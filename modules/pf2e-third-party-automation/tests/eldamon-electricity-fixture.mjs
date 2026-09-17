import {createElectricityLedger,ELECTRICITY_SOURCES as S} from '../scripts/eldamon-electricity.mjs';
const ID='pf2e-third-party-automation';
function fixture(){
 const docs=new Map(),gm={id:'gm',isGM:true},owner={id:'owner'},users=new Map([['gm',gm],['owner',owner]]);users.activeGM=gm;
 const game={user:gm,users,messages:new Map(),actors:new Map(),combats:new Map()},scene={id:'s',tokens:new Map()};let sequence=0;
 const set=(o,k,v)=>{const p=k.split('.');for(const part of p.slice(0,-1))o=o[part]??={};const key=p.at(-1);if(key.startsWith('-='))delete o[key.slice(2)];else o[key]=structuredClone(v);};
 function actor(id,alliance){const a={id,uuid:`Actor.${id}`,alliance,flags:{},items:new Map(),testUserPermission:u=>u===gm||u===owner,
  async update(d){for(const[k,v]of Object.entries(d))set(this,k,v);},
  async createEmbeddedDocuments(_type,entries){return entries.map(d=>item(a,`e${++sequence}`,d._stats.compendiumSource,d));},
  async deleteEmbeddedDocuments(_type,ids){for(const id of ids){const i=this.items.get(id);this.items.delete(id);if(i)docs.delete(i.uuid);for(const child of this.items.values())if(child.flags?.pf2e?.grantedBy?.id===id){this.items.delete(child.id);docs.delete(child.uuid);}}},
 };docs.set(a.uuid,a);game.actors.set(id,a);return a;}
 function item(a,id,source,data={}){const i={id,uuid:`${a.uuid}.Item.${id}`,sourceId:source,actor:a,type:'effect',flags:{},system:{},...structuredClone(data),
  async update(d){if(d['system.badge.value']===0)return a.deleteEmbeddedDocuments('Item',[id]);for(const[k,v]of Object.entries(d))set(this,k,v);}};a.items.set(id,i);docs.set(i.uuid,i);return i;}
 const caster=actor('caster','party'),target=actor('target','opposition'),other=actor('other','opposition'),outsider=actor('outsider','party');
 function token(a){const t={id:a.id,name:a.id,uuid:`Scene.s.Token.${a.id}`,documentName:'Token',actor:a,parent:scene,object:{distanceTo:()=>10}};scene.tokens.set(t.id,t);docs.set(t.uuid,t);return t;}
 const tokens=[caster,target,other,outsider].map(token),combat={id:'c',started:true,round:1,turn:0,turns:[{id:'casterturn',actor:caster,token:tokens[0]},{id:'targetturn',actor:target,token:tokens[1]}]};combat.combatants=combat.turns;game.combat=combat;game.combats.set(combat.id,combat);
 const templates={charged:{type:'effect',flags:{},system:{rules:[{key:'GrantItem',uuid:S.shocked}],badge:{type:'counter',value:1,max:3}}},shocked:{type:'effect',flags:{},system:{rules:[{key:'FlatModifier',selector:['fortitude','reflex'],value:-2}],duration:{unit:'unlimited'}}}};
 for(const key of ['charged','shocked'])docs.set(S[key],{type:'effect',toObject:()=>structuredClone(templates[key])});
 const power=item(caster,'power',S.surge,{type:'feat'}),receipt={nonce:'channel',status:'committed',sourceUuid:S.surge,actorUuid:caster.uuid,userId:owner.id,itemUuid:power.uuid,messageUuid:'ChatMessage.channel',powerId:'electric-surge',selection:{discharge:false,targetUuids:[tokens[1].uuid]},snapshot:{kind:'normal'}};
 const card={id:'channel',uuid:receipt.messageUuid,author:owner,speaker:{actor:caster.id,scene:'s',token:caster.id},flags:{pf2e:{origin:{uuid:power.uuid}},[ID]:{metapowerUse:{nonce:receipt.nonce,actorUuid:caster.uuid,itemUuid:power.uuid}}}};
 caster.flags[ID]={metapower:{receipts:{channel:receipt}}};game.messages.set(card.id,card);docs.set(card.uuid,card);
 const ledger=()=>createElectricityLedger({game,fromUuid:async id=>docs.get(id),reactionAvailable:()=>true});
 function source(kind='pure',targetUuids=[tokens[1].uuid]){
  const nonce=`source${++sequence}`,m={id:nonce,uuid:`ChatMessage.${nonce}`,rolls:[{options:{[ID]:{electricitySource:{nonce}}},instances:[{type:'electricity',total:13},...(kind==='mixed'?[{type:'slashing',total:8}]:[])]}],flags:{pf2e:{context:{type:'damage-roll'}},[ID]:{electricitySource:{nonce,effectKey:`channel:${caster.uuid}:channel`,targetUuids}}}};
  game.messages.set(m.id,m);docs.set(m.uuid,m);return m;
 }
 async function damage(a=target,{kind='pure',amount=13,m=source(kind),nonce=`application${++sequence}`}={}){
  const payload={nonce,actorUuid:a.uuid,tokenUuid:`Scene.s.Token.${a.id}`,sourceMessageUuid:m.uuid,sourceNonce:m.id,sourceItemUuid:power.uuid,rollIndex:0,kind,effectKey:m.flags[ID].electricitySource.effectKey};
  await ledger().beginDamage(payload,gm);
  const r={id:`receipt${++sequence}`,uuid:`ChatMessage.receipt${sequence}`,author:gm,speaker:{actor:a.id,scene:'s',token:a.id},flags:{pf2e:{context:{type:'damage-taken',options:[`${ID}:electricity-apply:${nonce}`]},origin:{uuid:power.uuid},appliedDamage:amount?{uuid:a.uuid,isHealing:false,updates:[{path:'system.attributes.hp.value',value:Math.min(4,amount)}]}:null},[ID]:{electricityApplied:{nonce,amount}}}};
  game.messages.set(r.id,r);docs.set(r.uuid,r);return {payload,receipt:r,finish:()=>ledger().finishDamage({actorUuid:a.uuid,nonce,receiptUuid:r.uuid},gm)};
 }
 return {docs,game,gm,owner,scene,combat,caster,target,other,outsider,tokens,power,receipt,card,ledger,item,source,damage,channel:{actorUuid:caster.uuid,nonce:receipt.nonce,messageUuid:card.uuid}};
}

export {fixture};
