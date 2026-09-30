import {MODULE_ID,clone,emptyLedger} from './schema.mjs';
export const isActiveGM=game=>!!game.user?.isGM&&game.users?.activeGM?.id===game.user.id;
export function createDocumentStore({game,JournalEntry,fromUuid}) {
  const authority=()=>{if(!isActiveGM(game))throw Error('active-gm-required')};
  const lookup=async()=>{const uuid=game.settings.get(MODULE_ID,'explorationLedgerUUID');return uuid?await fromUuid(uuid):null};
  return {
    read:async()=>{const journal=await lookup();return clone(journal?.getFlag(MODULE_ID,'explorationLedger')??emptyLedger())},
    write:async state=>{
      authority();let journal=await lookup();authority();
      if(!journal){
        journal=await JournalEntry.create({name:'探索活动账本',ownership:{default:0},flags:{[MODULE_ID]:{explorationLedger:emptyLedger()}}});
        authority();await game.settings.set(MODULE_ID,'explorationLedgerUUID',journal.uuid);authority();
      }
      await journal.setFlag(MODULE_ID,'explorationLedger',clone(state));authority();
    }
  };
}
