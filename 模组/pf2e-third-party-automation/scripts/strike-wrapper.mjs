import {MODULE_ID} from './rules.mjs';

const values=c=>Array.from(c?.values?.()??c??[]);
const strikePath='CONFIG.PF2E.Actor.documentClasses.character.prototype.prepareStrike';

/** The module's only character Strike wrapper. Native damage is tagged with the
 * attack card it follows, then wrapStrike lets each provider rewrite the Strike.
 * The bear-attack marker keeps its historical name: dual strikes and spell
 * combination write the same option, and its readers match that exact prefix. */
export function registerStrikeWrapper({game,libWrapper,wrapStrike}={}){
 if(libWrapper)libWrapper.register(MODULE_ID,strikePath,function(wrapped,...args){
  const strike=wrapped(...args),firstRoll=strike?.variants?.[0]?.roll;
  for(const method of ['damage','critical']){
   const native=strike?.[method];if(typeof native!=='function')continue;
   strike[method]=async function(params={}){
    // Native PF2e chat buttons and Workbench pass this exact context object.
    // No search by most recent weapon, name, actor, or target is permitted.
    const attack=params.checkContext&&values(game.messages).find(m=>m.isCheckRoll&&m.flags.pf2e.context===params.checkContext);
    const options=new Set(params.options??[]);
    if(attack)options.add(`${MODULE_ID}:bear-attack:${attack.id}`);
    return native.call(this,{...params,options});
   };
  }
  const result=wrapStrike?.(strike,this)??strike;
  if(typeof firstRoll==='function'&&typeof result?.variants?.[0]?.roll==='function')for(const alias of ['attack','roll'])if(result[alias]===firstRoll)result[alias]=result.variants[0].roll;
  return result;
 },'WRAPPER');
 return ()=>{if(libWrapper)libWrapper.unregister(MODULE_ID,strikePath);};
}
