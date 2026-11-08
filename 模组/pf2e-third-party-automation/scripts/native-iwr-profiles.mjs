/** Exact, independently audited PF2e 8.5 release bytes. They identify bridges
 * installed before descriptors named a source contract. Safe for browser imports. */
export const NATIVE_IWR_BRIDGE_PROTOCOL='pf2e-third-party-automation:iwr:2';
export const NATIVE_IWR_PROFILES=Object.freeze({
 '8.5.0':Object.freeze({
  systemVersion:'8.5.0',
  originalSHA256:'2929cbcc2e0c27e1e1a921c05b00263e55e113ba3210fcde379af9d2aa61e43d',
  patchedSHA256:'b4335fa31d1c7b36e100522e6fa3ced7c24075abb14b01a49f16a518e9fa4ec8',
  applyDamageSHA256:'fdd09b9ff0acfda43025fb9972e98143a4afa645e97bd6c49b9c4263139952a0',
  protocol:NATIVE_IWR_BRIDGE_PROTOCOL,
 }),
 '8.5.1':Object.freeze({
  systemVersion:'8.5.1',
  originalSHA256:'8fa38a2fcbf848ad967c75876ca33ebf5a46fb7d6a8cc90d8323c0fb5d471bc7',
  patchedSHA256:'d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157',
  sharedManualCompositionSHA256:'9be357c96dff3d0790edcb0d7889db98cfded0f41f34fd161233ea0bdd0dab11',
  applyDamageSHA256:'cd1d391b7d4c2c8f3b11b903c477a5e6e330343a94ba51dd5ddebbe610adcb5a',
  protocol:NATIVE_IWR_BRIDGE_PROTOCOL,
 }),
});

// A bridge descriptor names the audited applyDamage shape it was built under.
// The method is compared with NATIVE_DAMAGE_REFERENCE (native-source-shapes.mjs)
// by rename-equivalence; upstream bundle hashes and versions are recorded as
// evidence only. Earlier releases wrote one id per release's bytes of that same
// shape, and the files they installed still qualify by shape.
export const NATIVE_DAMAGE_CONTRACT='pf2e-apply-damage:shape:1';
const EARLIER_DAMAGE_CONTRACTS=['pf2e-apply-damage:D:2','pf2e-apply-damage:O:2','pf2e-apply-damage:O:1','pf2e-apply-damage:D:1'];
export const isNativeDamageContract=id=>id===NATIVE_DAMAGE_CONTRACT||EARLIER_DAMAGE_CONTRACTS.includes(id);
export const SHARED_MANUAL_SHAPE=Object.freeze({
 id:'pf2e-shared-manual:1',toolbeltId:'toolbelt-shared-manual:1',
 batchOriginal:'d139b4ab1e3c3ca72d5ca9a680f8adbc236c70959ca3c69ca6d6aa33118a696d',batchPatched:'3fb62d29f88fdfb530bd499f2f7e9ea1b29857081f5a4a3a417f53cbab582570',
 flatOriginal:'e0ff925c70305599bb26b5f3e4cf1c745510e84d31788b5f89abfc55b8f8d8dc',stacking:'c47f085d0b258d7ec4dadb37dfa7d68b17ff8e51d31af7173dc4605a99a37cba',
 nativeObserver:'aee30a4c318cc1645ef8bfcb00f53aec4438ed285048225ac0ca687af4d85a42',toolObserver:'43cb15acccea69cf8eea1e2ee50b5e3838358ea0aac70ee18eecf697f6659f98'
});
// PF2e 8.6.0 renames DamageRoll, Modifier and the API initializer; 8.6.1
// renames them again along with the troop iterator and reflows the stacking
// loop. Each row keeps those audited bindings together; unknown local changes
// still fail.
export const NATIVE_SHARED_SHAPES=Object.freeze([
 Object.freeze({batchOriginal:SHARED_MANUAL_SHAPE.batchOriginal,batchPatched:SHARED_MANUAL_SHAPE.batchPatched,flatOriginal:SHARED_MANUAL_SHAPE.flatOriginal,stacking:SHARED_MANUAL_SHAPE.stacking,init:'jm.onInit();',troopIterator:'gt',receiverRule:'Y',receiverPredicate:'Hn',nativeObserver:SHARED_MANUAL_SHAPE.nativeObserver}),
 Object.freeze({batchOriginal:'79954c84891ec4c9ced17ca2548de6c1e525e3e9d9bb6b0161c2096d35e642e3',batchPatched:'0e9b7b560205176460536a6ca8f3dd4699dfe08f3ab6cdc7cecdba2917bd8c03',flatOriginal:'f876b500e255635eb5bea585838a41e33fe8cabb38d62487402cab76217251dd',stacking:SHARED_MANUAL_SHAPE.stacking,init:'Nm.onInit();',troopIterator:'gt',receiverRule:'q',receiverPredicate:'Un',nativeObserver:'4af3b9ce7299921f61bb7e62335d5e2a1d55257d44449e5af41b844b4020e65d'}),
 Object.freeze({batchOriginal:'a700f08cb6de82fbc65fce4aa8202d2e315c5dbb648f8532fd1fe609c0927320',batchPatched:'d46e1ab50657083bc619b9e3cba3b7382e99413b2acf69680e95ff19ac2ac897',flatOriginal:'f876b500e255635eb5bea585838a41e33fe8cabb38d62487402cab76217251dd',stacking:'cf66f5bc249822044d6e5a483b25d3a8b592f7bb68a94e826ac798c58755f23d',init:'Fm.onInit();',troopIterator:'_t',receiverRule:'q',receiverPredicate:'Wn',nativeObserver:'35dd811d725745b1bab007d47ac1c67943e2da6c73a7e1266d739b0e8f57ee71'})
]);
const keys=(value,fields)=>Object.keys(value).sort().join(',')===fields.split(',').sort().join(',');
export function isAutomaticBatchDescriptor(value){return !!value&&keys(value,'version,protocol,providerId,providerVersion,baseSourceSHA256,sourceContract,model,staticReceiverModelVersion,receiverPredicateModelVersion')&&value.version===2&&value.protocol==='pf2e-third-party-automation:manual-pool-batch:1'&&value.providerId==='pf2e'&&typeof value.providerVersion==='string'&&/^[a-f0-9]{64}$/.test(value.baseSourceSHA256)&&value.sourceContract===SHARED_MANUAL_SHAPE.id&&value.model==='numeric-static-reception.v1'&&value.staticReceiverModelVersion===1&&value.receiverPredicateModelVersion===1}
export function isAutomaticToolDescriptor(value){return !!value&&keys(value,'version,hpBaselineGuardVersion,providerId,providerVersion,sourceSHA256,sourceContract')&&value.version===2&&value.hpBaselineGuardVersion===1&&value.providerId==='pf2e-toolbelt'&&typeof value.providerVersion==='string'&&/^[a-f0-9]{64}$/.test(value.sourceSHA256)&&value.sourceContract===SHARED_MANUAL_SHAPE.toolbeltId}
