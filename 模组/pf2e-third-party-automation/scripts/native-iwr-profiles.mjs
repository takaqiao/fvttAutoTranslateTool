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
  applyDamageSHA256:'cd1d391b7d4c2c8f3b11b903c477a5e6e330343a94ba51dd5ddebbe610adcb5a',
  protocol:NATIVE_IWR_BRIDGE_PROTOCOL,
  callbackSignature:'?.nativeDamageIWR?.(this, arguments[0], f, r, { actorDamage: x - O - ie, shieldDamage: te })',
 }),
});
