import {getSourceId} from './native-context.mjs';

export const VOLATILE_SPARK_SOURCES=Object.freeze({
 study:'Compendium.pf2e.classfeatures.Item.aRDNtB0mh6MMRSKf',
 booming:'Compendium.pf2e.spells-srd.Item.DNvfKfDSs9DstDfT',
 reload:'Compendium.pf2e.actionspf2e.Item.tmbf6xouEnRXCnKR',
 cascade:'Compendium.pf2e.actionspf2e.Item.HbejhIywqIufrmVM',
});
export const hasVolatileSpark=actor=>Array.from(actor?.items?.values?.()??actor?.items??[]).some(item=>item.type==='feat'&&!item.suppressed&&!item.isSuppressed&&!item.system?.suppressed&&getSourceId(item)===VOLATILE_SPARK_SOURCES.study);
export const heldFirearm=strike=>strike?.type==='strike'&&strike.ready!==false&&strike.item?.type==='weapon'&&strike.item.isRanged===true&&(strike.item.group??strike.item.system.group)==='firearm'&&strike.item.system.equipped?.carryType==='held'&&strike.item.system.equipped.handsHeld>0;
