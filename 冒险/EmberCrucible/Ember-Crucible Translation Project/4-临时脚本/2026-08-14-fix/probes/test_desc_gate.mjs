// 用假的 schema 复刻 crucible 的两种 description 形状，验证新判据的取舍是否正确。
// 这是对 register.js 里 descriptionExpectsObject() 的 specificity + sensitivity 双向回测。
class DataField {}
class SchemaField extends DataField { constructor(fields){ super(); this.fields = fields; } }
class StringField extends DataField {}
class HTMLField extends StringField {}
globalThis.foundry = { data: { fields: { SchemaField, StringField, HTMLField } } };

// crucible 0.10.1 实况：只有 physical 是 SchemaField，其余十类是 HTMLField
const MODELS = {
  weapon:    { schema: { fields: { description: new SchemaField({ public: new HTMLField(), private: new HTMLField() }) } } },
  armor:     { schema: { fields: { description: new SchemaField({ public: new HTMLField(), private: new HTMLField() }) } } },
  talent:    { schema: { fields: { description: new HTMLField() } } },
  spell:     { schema: { fields: { description: new HTMLField() } } },
  ancestry:  { schema: { fields: { description: new HTMLField() } } },
  archetype: { schema: { fields: { description: new HTMLField() } } },
  background:{ schema: { fields: { description: new HTMLField() } } },
  taxonomy:  { schema: { fields: { description: new HTMLField() } } },
  loot:      { schema: { fields: { description: new HTMLField() } } },
  schematic: { schema: { fields: { description: new HTMLField() } } },
};
globalThis.CONFIG = { Item: { dataModels: MODELS } };

// ── 被测函数（与 register.js 逐字一致）────────────────────────────
function descriptionExpectsObject(doc) {
  if (!doc) return false;
  let field = doc?.system?.schema?.fields?.description;
  if (!field && typeof doc.type === 'string') {
    const model = globalThis.CONFIG?.Item?.dataModels?.[doc.type];
    field = model?.schema?.fields?.description;
  }
  if (!field) return false;
  const SF = foundry?.data?.fields?.SchemaField;
  if (SF && field instanceof SF) return true;
  return !!field?.fields?.public;
}

const mkDoc = (type) => ({ type, system: { schema: MODELS[type].schema } });
let pass = 0, fail = 0;
const check = (name, got, want) => {
  if (got === want) { pass++; }
  else { fail++; console.log(`  ✗ ${name}: got ${got}, want ${want}`); }
};

console.log('— 特异度：十类 HTMLField 必须一律不转（否则写成 "[object Object]"）—');
for (const t of ['talent','spell','ancestry','archetype','background','taxonomy','loot','schematic'])
  check(`实例 ${t}`, descriptionExpectsObject(mkDoc(t)), false);
for (const t of ['talent','spell','taxonomy'])
  check(`裸载荷 {type:${t}}`, descriptionExpectsObject({ type: t }), false);

console.log('— 灵敏度：physical 必须仍然转（真目标不能漏）—');
for (const t of ['weapon','armor']) {
  check(`实例 ${t}`, descriptionExpectsObject(mkDoc(t)), true);
  check(`裸载荷 {type:${t}}`, descriptionExpectsObject({ type: t }), true);
}

console.log('— 兜底：问不出 schema 时必须返回 false（宁可漏修不可写坏）—');
check('null', descriptionExpectsObject(null), false);
check('未知 type', descriptionExpectsObject({ type: 'nosuchtype' }), false);
check('无 type 无 system', descriptionExpectsObject({ foo: 1 }), false);
check('跨 realm（instanceof 失效）', (() => {
  const alien = { fields: { public: {}, private: {} } };           // 不是我们这个 realm 的 SchemaField
  return descriptionExpectsObject({ system: { schema: { fields: { description: alien } } } });
})(), true);

console.log(`\n通过 ${pass} / 失败 ${fail}`);
process.exit(fail ? 1 : 0);
