/** Exact, independently audited PF2e release bytes. Safe for browser imports. */
export const NATIVE_IWR_BRIDGE_PROTOCOL='pf2e-third-party-automation:iwr:2';
export const NATIVE_IWR_PROFILES=Object.freeze({
 '8.5.0':Object.freeze({
  systemVersion:'8.5.0',
  originalSHA256:'2929cbcc2e0c27e1e1a921c05b00263e55e113ba3210fcde379af9d2aa61e43d',
  patchedSHA256:'b4335fa31d1c7b36e100522e6fa3ced7c24075abb14b01a49f16a518e9fa4ec8',
  applyDamageSHA256:'fdd09b9ff0acfda43025fb9972e98143a4afa645e97bd6c49b9c4263139952a0',
  protocol:NATIVE_IWR_BRIDGE_PROTOCOL,
  callbackSignature:'?.nativeDamageIWR?.(this, arguments[0], f, r, { actorDamage: x - D - ie, shieldDamage: ne })',
 }),
 '8.5.1':Object.freeze({
  systemVersion:'8.5.1',
  originalSHA256:'8fa38a2fcbf848ad967c75876ca33ebf5a46fb7d6a8cc90d8323c0fb5d471bc7',
  patchedSHA256:'d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157',
  sharedManualCompositionSHA256:'9be357c96dff3d0790edcb0d7889db98cfded0f41f34fd161233ea0bdd0dab11',
  applyDamageSHA256:'cd1d391b7d4c2c8f3b11b903c477a5e6e330343a94ba51dd5ddebbe610adcb5a',
  protocol:NATIVE_IWR_BRIDGE_PROTOCOL,
  callbackSignature:'?.nativeDamageIWR?.(this, arguments[0], f, r, { actorDamage: x - O - ie, shieldDamage: te })',
 }),
});

// These contracts cover local native semantics; upstream bundle hashes and
// versions remain diagnostics on the automatic source path.
export const NATIVE_DAMAGE_SHAPES=Object.freeze([
 Object.freeze({id:'pf2e-apply-damage:O:2',originalSHA256:'3f4ef8c14ff22bb2c8722648a68f5c84d3b48aeb8aa9cd1bc00b78e7cf61780a',applyDamageSHA256:'e1d69d11d7a4202673bd432a0f9780d082c9e8f83941910f95da6bc0db955a00',callbackSignature:NATIVE_IWR_PROFILES['8.5.1'].callbackSignature,shield:'O'}),
 Object.freeze({id:'pf2e-apply-damage:O:1',originalSHA256:'de3c5091f79733be48a87e504ecb0c030ad882e72cd336c7a83f9f92476d0feb',applyDamageSHA256:NATIVE_IWR_PROFILES['8.5.1'].applyDamageSHA256,callbackSignature:NATIVE_IWR_PROFILES['8.5.1'].callbackSignature,shield:'O'}),
 Object.freeze({id:'pf2e-apply-damage:D:1',originalSHA256:'d21ac5f5ead5f00025de8cd66586e77489b474f444c5273382ec0e305bc3f741',applyDamageSHA256:NATIVE_IWR_PROFILES['8.5.0'].applyDamageSHA256,callbackSignature:NATIVE_IWR_PROFILES['8.5.0'].callbackSignature,shield:'D'})
]);
export const SHARED_MANUAL_SHAPE=Object.freeze({
 id:'pf2e-shared-manual:1',toolbeltId:'toolbelt-shared-manual:1',
 batchOriginal:'d139b4ab1e3c3ca72d5ca9a680f8adbc236c70959ca3c69ca6d6aa33118a696d',batchPatched:'3fb62d29f88fdfb530bd499f2f7e9ea1b29857081f5a4a3a417f53cbab582570',
 flatOriginal:'e0ff925c70305599bb26b5f3e4cf1c745510e84d31788b5f89abfc55b8f8d8dc',stacking:'c47f085d0b258d7ec4dadb37dfa7d68b17ff8e51d31af7173dc4605a99a37cba',
 toolReceive:'10ff0e183740c1f5f35d604c6cdb457b3e94eb1b0fd3343ada7acf7f1403466a',toolForward:'c593ae1650f9c6b726eec7735539db6ae8587b77359c72fd71e13fa26504f4d1',
 toolSocket:'326a59c68139226442b5c9cc441a8e9e2a269efa1ff95b5dc5c57ef2b8411b7a',toolValidity:'cd6440b56acc125dc596a867fde21109dfc1dd5b18f0572d7a2a42c4b2654d90',toolBinding:'4e7df87b048f39bd9f4ffa641f92c913c31f1ad36b93699779e7a8e22e7c3e04',
 nativeObserver:'aee30a4c318cc1645ef8bfcb00f53aec4438ed285048225ac0ca687af4d85a42',toolObserver:'43cb15acccea69cf8eea1e2ee50b5e3838358ea0aac70ee18eecf697f6659f98'
});
// PF2e 8.6.0 renames DamageRoll, Modifier and the API initializer. Each row
// keeps those audited bindings together; unknown local changes still fail.
export const NATIVE_SHARED_SHAPES=Object.freeze([
 Object.freeze({batchOriginal:SHARED_MANUAL_SHAPE.batchOriginal,batchPatched:SHARED_MANUAL_SHAPE.batchPatched,flatOriginal:SHARED_MANUAL_SHAPE.flatOriginal,init:'jm.onInit();',receiverRule:'Y',receiverPredicate:'Hn',nativeObserver:SHARED_MANUAL_SHAPE.nativeObserver}),
 Object.freeze({batchOriginal:'79954c84891ec4c9ced17ca2548de6c1e525e3e9d9bb6b0161c2096d35e642e3',batchPatched:'0e9b7b560205176460536a6ca8f3dd4699dfe08f3ab6cdc7cecdba2917bd8c03',flatOriginal:'f876b500e255635eb5bea585838a41e33fe8cabb38d62487402cab76217251dd',init:'Nm.onInit();',receiverRule:'q',receiverPredicate:'Un',nativeObserver:'4af3b9ce7299921f61bb7e62335d5e2a1d55257d44449e5af41b844b4020e65d'})
]);
const keys=(value,fields)=>Object.keys(value).sort().join(',')===fields.split(',').sort().join(',');
// The same authenticated sender contract with audited minifier bindings.
export const TOOL_SOCKET_CONTRACTS=Object.freeze([
 Object.freeze({socket:SHARED_MANUAL_SHAPE.toolSocket,transport:'b5d5ec6227f0c4c03c6147276e9b8681ba238855c7b0e64f3077cd7ccb5f7622',conversion:'1225b9adf3c1269012c1579001a08eff161785cca0c7dd263b95857a6f7fe72d'}),
 Object.freeze({socket:'fe97a3e4a20c6ce1b297e7f3b62d225632833e09ccd4e43f82c6160864e25b44',transport:'ad19fcef9658d821cbe27d72d38b4a15f8ab0f66aab52fa556881fddc6efc5eb',conversion:'186abafaa3017dd447f7d98a9ca4346c542efea9387f37cae5e284221e672679'}),
 Object.freeze({socket:'a763ce82944d90f2895455895000e898e85234fe8a1d03aaf3b55368676c5c1f',transport:'f3fa49300b4cb4946a522d0355bbc24e9bf7a0198d2c111b069357d5548558ce',conversion:'f7aac9d8e32c74636a756656efd1f3670162d188a2c81add1e975f30cadfe2c2',binding:'12ccbbd4a64b2e6f617154c6b79f4cd6ef800cb1a87750d14de01f4e62e0de5a'})
]);
export function isAutomaticBatchDescriptor(value){return !!value&&keys(value,'version,protocol,providerId,providerVersion,baseSourceSHA256,sourceContract,model,staticReceiverModelVersion,receiverPredicateModelVersion')&&value.version===2&&value.protocol==='pf2e-third-party-automation:manual-pool-batch:1'&&value.providerId==='pf2e'&&typeof value.providerVersion==='string'&&/^[a-f0-9]{64}$/.test(value.baseSourceSHA256)&&value.sourceContract===SHARED_MANUAL_SHAPE.id&&value.model==='numeric-static-reception.v1'&&value.staticReceiverModelVersion===1&&value.receiverPredicateModelVersion===1}
export function isAutomaticToolDescriptor(value){return !!value&&keys(value,'version,hpBaselineGuardVersion,providerId,providerVersion,sourceSHA256,sourceContract')&&value.version===2&&value.hpBaselineGuardVersion===1&&value.providerId==='pf2e-toolbelt'&&typeof value.providerVersion==='string'&&/^[a-f0-9]{64}$/.test(value.sourceSHA256)&&value.sourceContract===SHARED_MANUAL_SHAPE.toolbeltId}
