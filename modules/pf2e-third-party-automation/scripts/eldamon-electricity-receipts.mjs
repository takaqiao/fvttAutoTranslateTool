const values=collection=>Array.from(collection?.values?.()??collection??[]);
/** A projection of current APPLY-tagged native receipts, never authority for
 * amount, ownership or settlement. The ledger still checks the whole receipt. */
export function createElectricityReceiptIndex({game,prefix}){
 const buckets=new Map(),documents=new Map();let seeded=false;
 const nonceOf=message=>{
  if(message?.flags?.pf2e?.context?.type!=='damage-taken')return null;
  const tags=(message.flags.pf2e.context.options??[]).filter(option=>typeof option==='string'&&option.startsWith(prefix));
  return tags.length===1?tags[0].slice(prefix.length):null;
 };
 function remove(id){
  const previous=documents.get(id);if(!previous)return;
  const bucket=buckets.get(previous.nonce);bucket?.delete(previous.message);
  if(bucket&&!bucket.size)buckets.delete(previous.nonce);documents.delete(id);
 }
 function remember(message){
  if(!seeded||!message?.id||game.messages.get(message.id)!==message)return;
  remove(message.id);const nonce=nonceOf(message);if(!nonce)return;
  let bucket=buckets.get(nonce);if(!bucket)buckets.set(nonce,bucket=new Set());
  bucket.add(message);documents.set(message.id,{message,nonce});
 }
 function forget(message){if(documents.get(message?.id)?.message===message)remove(message.id)}
 function seed(){if(seeded)return;seeded=true;for(const message of values(game.messages))remember(message)}
 return {seed,remember,forget,values(nonce){seed();return [...buckets.get(nonce)??[]].filter(message=>game.messages.get(message.id)===message&&nonceOf(message)===nonce)}};
}
