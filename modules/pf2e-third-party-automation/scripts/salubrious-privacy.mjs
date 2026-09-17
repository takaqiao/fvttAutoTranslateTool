// Audience data are identities, never a copy of a secret roll or diagnosis.
// Persist the originating user's audience; applyMode('self') on the later GM
// would select the wrong recipient.
const modes=new Set(['public','gm','blind','self']);
const list=collection=>Array.from(collection?.values?.()??collection??[]);
const ids=value=>[...new Set(value??[])].sort();
const equal=(a,b)=>JSON.stringify(ids(a))===JSON.stringify(ids(b));
const invalid=reason=>{throw Error(`Invalid salubrious privacy: ${reason}`)};
const normalize=mode=>mode==='ic'?'public':mode;
const isSecret=(item,context)=>item?.system?.traits?.value?.includes('secret')||new Set(context?.options??[]).has('secret')||context?.traits?.some(t=>(t.name??t)==='secret');
const floor=(mode,user,restricted)=>!restricted?mode:user.isGM?(['blind','self'].includes(mode)?mode:'gm'):'blind';
export function captureSalubriousPrivacy({game,user=game.user,token,item,requestedMode=game.settings?.get('core','messageMode')}){
 const requested=normalize(requestedMode);if(!modes.has(requested))invalid('unknown mode');
 if(game.users.get(user?.id)!==user||!user.active)invalid('inactive source user');
 const mode=floor(requested,user,token?.hidden||isSecret(item));
 const whisper=mode==='public'?[]:mode==='self'?[user.id]:list(game.users).filter(u=>u.isGM).map(u=>u.id);
 if(mode!=='public'&&!whisper.length)invalid('empty private audience');
 return {schema:1,userId:user.id,mode,whisper:ids(whisper),blind:mode==='blind'};
}
export function sameSalubriousPrivacy(a,b){
 if(a==null||b==null)return a==null&&b==null;
 return ['schema','userId','mode','blind'].every(k=>a[k]===b[k])&&equal(a.whisper,b.whisper);
}
export function treatmentPrivacyForPatient({game,user,token,item,target,privacy}){
 if(!privacy)return undefined;
 salubriousPrivacyData({privacy,userId:user.id});
 const restricted=token?.hidden||isSecret(item)||target?.hidden||game.settings?.get('pf2e','metagame_secretDamage')&&!target?.actor?.hasPlayerOwner;
 const mode=floor(privacy.mode,user,restricted);if(mode===privacy.mode)return structuredClone(privacy);
 return captureSalubriousPrivacy({game,user,token,item,requestedMode:mode});
}
export function mergeSalubriousAudience(privacy,native){
 const source=salubriousPrivacyData({privacy,userId:privacy.userId}),before=ids(native?.whisper??[]);
 const whisper=source.whisper.length?(before.length?source.whisper.filter(id=>before.includes(id)):source.whisper):before;
 if(source.whisper.length&&before.length&&!whisper.length)invalid('empty native/source audience intersection');
 const blind=source.blind||!!native?.blind,mode=blind?'blind':!whisper.length?'public':equal(whisper,[privacy.userId])?'self':'gm';
 return {...privacy,mode,whisper:ids(whisper),blind};
}
export function assertSalubriousReceiptPrivacy({message,claim}){
 const p=message?.flags?.['pf2e-third-party-automation']?.salubriousKiss?.privacy,actual=salubriousPrivacyData({privacy:p,userId:claim.userId}),source=salubriousPrivacyData(claim);
 if(!p||source.blind&&!actual.blind||source.whisper.length&&(!actual.whisper.length||actual.whisper.some(id=>!source.whisper.includes(id))))invalid('native receipt audience widened');
 return assertSalubriousCardPrivacy({message,claim:{...claim,privacy:p}});
}
export function salubriousPrivacyData(claim){
 const p=claim?.privacy;
 if(!p)return {blind:false,whisper:[]}; // frozen C's visible Player records
 if(Object.keys(p).some(k=>!['schema','userId','mode','whisper','blind'].includes(k))||p.schema!==1||p.userId!==claim.userId||!modes.has(p.mode)||p.blind!==(p.mode==='blind')||!Array.isArray(p.whisper)||p.whisper.some(id=>typeof id!=='string'||!id)||p.whisper.length!==ids(p.whisper).length)invalid('malformed audience');
 if(p.mode==='public'&&p.whisper.length||p.mode!=='public'&&!p.whisper.length||p.mode==='self'&&!equal(p.whisper,[claim.userId]))invalid('wrong audience');
 return {blind:p.blind,whisper:[...p.whisper]};
}
export function validateSalubriousPrivacy({game,user,token,item,privacy,context}){
 if(!privacy){if(user?.isGM||token?.hidden||isSecret(item,context))invalid('legacy public source cannot be restrictive');return 'public'}
 salubriousPrivacyData({privacy,userId:user.id});
 if(game.users.get(user?.id)!==user||!user.active||privacy.userId!==user.id)invalid('source user changed');
 if(floor(privacy.mode,user,token?.hidden||isSecret(item,context))!==privacy.mode)invalid('source became more restrictive');
 if(['gm','blind'].includes(privacy.mode)&&privacy.whisper.some(id=>!game.users.get(id)?.isGM))invalid('GM audience changed');
 return privacy.mode;
}
export function assertSalubriousCardPrivacy({message,claim}){
 const expected=salubriousPrivacyData(claim),context=message?.flags?.pf2e?.context,mode=claim.privacy?.mode??'public';
 if(context?.messageMode!==mode||!!message.blind!==expected.blind||!equal(message.whisper,expected.whisper))invalid('native card audience differs');
 if(mode==='public'&&new Set(context.options??[]).has('secret'))invalid('secret public card');
 return true;
}
