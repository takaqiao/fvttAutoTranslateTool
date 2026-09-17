import {GLIMPSE_SOURCES as S} from '../scripts/glimpse-source.mjs';
export function fixture(){
 const user={id:'u',active:true,isGM:true},users=new Map([['u',user]]);users.activeGM=user;
 const actor=(id,type)=>({id,uuid:`Actor.${id}`,type,items:new Map(),canAct:true,isDead:false,level:5,system:{traits:{value:[]}},testUserPermission:u=>u===user,isEnemyOf:a=>a.id==='enemy',isAllyOf:a=>a.id==='ally'});
 const champion=actor('champion','character'),enemy=actor('enemy','npc'),ally=actor('ally','character'),actors=new Map([champion,enemy,ally].map(a=>[a.id,a]));
 const scene={id:'scene',tokens:new Map()},token=a=>{const t={id:a.id,uuid:`Scene.scene.Token.${a.id}`,documentName:'Token',actor:a,parent:scene,auras:new Map()};t.object={document:t};scene.tokens.set(t.id,t);return t;};
 const championToken=token(champion),enemyToken=token(enemy),allyToken=token(ally);championToken.auras.set('champions-aura',{radius:15,containsToken:()=>true});
 const ability={id:'glimpse',uuid:`${champion.uuid}.Item.glimpse`,type:'action',sourceId:S.glimpse,actor:champion,system:{actionType:{value:'reaction'}}};champion.items.set(ability.id,ability);champion.items.set('aura',{id:'aura',type:'feat',sourceId:S.aura,actor:champion});
 const item={id:'attack',uuid:`${enemy.uuid}.Item.attack`,type:'melee',actor:enemy};enemy.items.set(item.id,item);
 const combat={id:'combat',started:true,round:2,turn:0,turns:[enemyToken,championToken,allyToken].map(t=>({id:t.id,actor:t.actor,token:t,flags:{}}))};
 const roll={total:10,options:{},instances:[{type:'slashing',persistent:false,total:10}],toJSON(){return {class:'DamageRoll',evaluated:true,formula:'{10[slashing]}',total:this.total};}};
 const message={id:'damage',uuid:'ChatMessage.damage',isDamageRoll:true,author:user,speaker:{actor:enemy.id,scene:scene.id,token:enemyToken.id},rolls:[roll],flags:{pf2e:{context:{type:'damage-roll',sourceType:'attack',target:{actor:ally.uuid,token:allyToken.uuid}},origin:{type:item.type,uuid:item.uuid,actor:enemy.uuid},strike:{damaging:true,actor:enemy.uuid}}}};
 const game={user,users,actors,scenes:new Map([[scene.id,scene]]),combats:new Map([[combat.id,combat]]),combat:{id:'unrelated',started:true,turns:[]},messages:new Map([[message.id,message]])};
 const docs=new Map([champion,enemy,ally,ability,item,championToken,enemyToken,allyToken,message].map(d=>[d.uuid,d]));const fromUuid=async uuid=>docs.get(uuid),params={damage:roll,token:allyToken,item,rollOptions:new Set()},source={messageId:message.id,rollIndex:0};
 return {game,user,champion,enemy,ally,ability,item,championToken,enemyToken,allyToken,scene,combat,roll,message,docs,fromUuid,params,source};
}