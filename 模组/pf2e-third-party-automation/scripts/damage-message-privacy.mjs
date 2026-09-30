/** Merged damage can only be shown to recipients allowed by every private part. */
export function mergeDamageMessagePrivacy(parts){
 let recipients=null,blind=false;
 for(const part of parts){
  if(!part)continue;const whisper=[...new Set(part.whisper??[])];blind||=part.blind===true;
  if(part.blind&&!whisper.length)throw Error('秘骰伤害缺少可见受众，不能公开合并。');
  if(!whisper.length)continue;
  recipients=recipients===null?whisper:recipients.filter(id=>whisper.includes(id));
  if(!recipients.length)throw Error('伤害部分没有共同可见受众，不能公开合并。');
 }
 return {blind,whisper:recipients??[]};
}
