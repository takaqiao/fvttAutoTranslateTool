import test from 'node:test';
import assert from 'node:assert/strict';
import {collectModuleUsage} from '../../scripts/lib/module-usage.mjs';

test('distinguishes rule references, module flags and content sources on the same item', () => {
  const doc = {
    _id: 'actor', name: '角色', items: [{
      _id: 'feat', name: '专长',
      _stats: {compendiumSource: 'Compendium.example.feats.Item.feat'},
      flags: {example: {choice: 'one'}},
      system: {rules: [{key: 'GrantItem', uuid: 'Compendium.example.effects.Item.effect'}]},
    }],
  };
  assert.deepEqual(collectModuleUsage(doc, [{id: 'example', title: '示例模块'}]), [{
    id: 'example', title: '示例模块', evidence: [
      {kind: 'flag', documentId: 'feat', documentName: '专长', path: 'items.0.flags.example'},
      {kind: 'reference', documentId: 'feat', documentName: '专长', path: 'items.0.system.rules.0.uuid'},
      {kind: 'source', documentId: 'feat', documentName: '专长', path: 'items.0._stats.compendiumSource'},
    ],
  }]);
});

for (const [rule, paths] of [
  [{key: 'GrantItem', uuid: 'Compendium.example.items.Item.feat'}, ['uuid']],
  [{key: 'ChoiceSet', rollOption: 'chosen-feat', choices: [
    {value: 'Compendium.example.items.Item.feat', label: '专长'},
    {value: 'Compendium.example.items.Actor.actor', label: '角色'},
    {value: 'Compendium.example.items.Item.other', label: '另一个专长'},
    {value: 'handwritten-slug', label: '手写选项'},
    {value: 'Compendium.legacy.pack.choice', label: '未确认 pack 类型的旧选项'},
  ]}, ['choices.0.value', 'choices.2.value']],
  [{key: 'Aura', effects: [{uuid: 'Compendium.example.items.Item.effect'}]}, ['effects.0.uuid']],
  [{key: 'EphemeralEffect', uuid: 'Compendium.example.items.Item.effect'}, ['uuid']],
  [{key: 'SpecialResource', itemUUID: 'Compendium.example.items.Item.resource'}, ['itemUUID']],
]) {
  test(`collects the supported ${rule.key} reference fields`, () => {
    const doc = {_id: 'item', name: '条目', system: {rules: [rule]}};
    assert.deepEqual(collectModuleUsage(doc), [{
      id: 'example', title: 'example', evidence: paths.map(path => ({
        kind: 'reference', documentId: 'item', documentName: '条目', path: `system.rules.0.${path}`,
      })),
    }]);
  });
}

test('collects ChoiceSet module references when every option is an explicit Item UUID', () => {
  const doc = {_id: 'item', name: '选择条目', system: {rules: [{key: 'ChoiceSet', choices: [
    {value: 'Compendium.example.items.Item.feat', label: '模块条目'},
    {value: 'Compendium.pf2e.feats-srd.Item.systemFeat', label: '系统条目'},
    {value: 'Item.worldItem', label: '世界条目'},
  ]}]}};
  assert.deepEqual(collectModuleUsage(doc), [{
    id: 'example', title: 'example', evidence: [
      {kind: 'reference', documentId: 'item', documentName: '选择条目', path: 'system.rules.0.choices.0.value'},
    ],
  }]);
});

for (const [kind, value] of [
  ['embedded Item', 'Actor.actorId.Item.itemId'],
  ['legacy Compendium', 'Compendium.pf2e.feats-srd.legacyFeat'],
]) {
  test(`explicit module Item references survive ${kind} choices`, () => {
    const doc = {_id: 'item', name: '选择条目', system: {rules: [{key: 'ChoiceSet', choices: [
      {value: 'Compendium.addon.features.Item.moduleFeat', label: '模块条目'},
      {value, label: '另一条目'},
    ]}]}};
    assert.deepEqual(collectModuleUsage(doc), [{
      id: 'addon', title: 'addon', evidence: [
        {kind: 'reference', documentId: 'item', documentName: '选择条目', path: 'system.rules.0.choices.0.value'},
      ],
    }]);
  });
}

test('unconditional non-Item choices suppress module references without a string rollOption', () => {
  for (const value of [42, null, 'Actor.actorId', 'Compendium.example.actors.Actor.actorId', 'JournalEntry.entryId']) {
    const doc = {_id: 'item', name: '选择条目', system: {rules: [{key: 'ChoiceSet', choices: [
      {value: 'Compendium.addon.features.Item.moduleFeat', label: '模块条目'},
      {value, label: '非 Item 选项'},
    ]}]}};
    assert.deepEqual(collectModuleUsage(doc), []);
  }
});

test('does not infer ChoiceSet references when a plain option remains and rollOption is not a string', () => {
  for (const selection of ['plain', 'Compendium.example.items.Item.feat']) {
    for (const plain of [{value: 'plain', label: 'Plain option'}, {value: 'plain', label: 'Plain option', predicate: []}]) {
      const rule = {
        key: 'ChoiceSet', flag: 'choice', selection, adjustName: false,
        choices: [{value: 'Compendium.example.items.Item.feat', label: 'Item-like string'}, plain],
      };
      assert.deepEqual(collectModuleUsage({_id: 'item', name: '选择条目', system: {rules: [rule]}}), []);
      rule.rollOption = true;
      assert.deepEqual(collectModuleUsage({_id: 'item', name: '选择条目', system: {rules: [rule]}}), []);
    }
  }
});

test('keeps possible ChoiceSet references when predicates can remove every non-Item option', () => {
  const doc = {_id: 'item', name: '选择条目', system: {rules: [{key: 'ChoiceSet', choices: [
    {value: 'Compendium.example.items.Item.feat', label: '模块条目'},
    {value: 'plain', label: '条件选项', predicate: ['trained:crafting']},
  ]}]}};
  assert.deepEqual(collectModuleUsage(doc), [{
    id: 'example', title: 'example', evidence: [
      {kind: 'reference', documentId: 'item', documentName: '选择条目', path: 'system.rules.0.choices.0.value'},
    ],
  }]);
});

test('attributes modern and legacy sources to actors, embedded items and effects', () => {
  const doc = {
    _id: 'actor', name: '角色', _stats: {compendiumSource: 'Compendium.actors.characters.Actor.actor'},
    items: [{
      _id: 'item', name: '条目', flags: {core: {sourceId: 'Compendium.items.feats.item'}},
      effects: [{_id: 'effect', name: '条目效果', flags: {core: {sourceId: 'Compendium.effects.effects.ActiveEffect.effect'}}}],
    }],
    effects: [{_id: 'effect2', name: '角色效果', _stats: {compendiumSource: 'Compendium.effects.effects.ActiveEffect.effect2'}}],
  };
  assert.deepEqual(collectModuleUsage(doc), [
    {id: 'actors', title: 'actors', evidence: [
      {kind: 'source', documentId: 'actor', documentName: '角色', path: '_stats.compendiumSource'},
    ]},
    {id: 'effects', title: 'effects', evidence: [
      {kind: 'source', documentId: 'effect', documentName: '条目效果', path: 'items.0.effects.0.flags.core.sourceId'},
      {kind: 'source', documentId: 'effect2', documentName: '角色效果', path: 'effects.0._stats.compendiumSource'},
    ]},
    {id: 'items', title: 'items', evidence: [
      {kind: 'source', documentId: 'item', documentName: '条目', path: 'items.0.flags.core.sourceId'},
    ]},
  ]);
});

test('only known module flag namespaces produce flag evidence', () => {
  const doc = {
    _id: 'actor', name: '角色', flags: {unknown: {uuid: 'Compendium.unknown.items.Item.foo'}, known: {enabled: true}},
    effects: [{_id: 'effect', name: '效果', flags: {known: {value: 1}}}],
  };
  const catalog = new Map([['known', {id: 'known', title: '已知模块'}]]);
  assert.deepEqual(collectModuleUsage(doc, catalog.values()), [{
    id: 'known', title: '已知模块', evidence: [
      {kind: 'flag', documentId: 'actor', documentName: '角色', path: 'flags.known'},
      {kind: 'flag', documentId: 'effect', documentName: '效果', path: 'effects.0.flags.known'},
    ],
  }]);
});

test('reuses frozen usage only as module names, without retaining old evidence', () => {
  const previous = [{id: 'example', title: '冻结名称', evidence: [
    {kind: 'reference', documentId: 'removed', documentName: '已移除', path: 'items.0.system.rules.0.uuid'},
  ]}];
  assert.deepEqual(collectModuleUsage({_id: 'actor', name: '角色', flags: {example: {enabled: true}}}, previous), [{
    id: 'example', title: '冻结名称', evidence: [
      {kind: 'flag', documentId: 'actor', documentName: '角色', path: 'flags.example'},
    ],
  }]);
  assert.deepEqual(collectModuleUsage({_id: 'actor', name: '角色'}, previous), []);
});

test('does not infer modules from names, prose, media or unrelated rule fields', () => {
  const uuid = 'Compendium.example.items.Item.feat';
  const doc = {
    _id: 'actor', name: 'Example 角色', img: 'modules/example/portrait.webp',
    system: {description: {value: `<p>@UUID[${uuid}]{说明}</p>`}},
    prototypeToken: {texture: {src: 'modules/example/token.webp'}},
    items: [{_id: 'item', name: 'Example 专长', system: {
      slug: 'example-feat', description: {value: `@UUID[${uuid}]`},
      rules: [
        {key: 'ItemAlteration', itemId: uuid, value: uuid},
        {key: 'RollOption', uuid},
        {key: 'GrantItem', uuid: `before ${uuid}`},
        {key: 'GrantItem', uuid: '{item|flags.example.choice}'},
        {key: 'ChoiceSet', choices: {filter: uuid}},
      ],
    }}],
  };
  assert.deepEqual(collectModuleUsage(doc, [{id: 'example', title: 'Example'}]), []);
});

test('excludes core, PF2e, world and this module from every evidence kind', () => {
  const excluded = ['core', 'pf2e', 'world', 'taka-charsync'];
  const doc = {_id: 'actor', name: '角色', items: excluded.map(id => ({
    _id: id, name: id,
    _stats: {compendiumSource: `Compendium.${id}.items.Item.feat`},
    flags: {[id]: {active: true}},
    system: {rules: [{key: 'GrantItem', uuid: `Compendium.${id}.items.Item.other`}]},
  }))};
  assert.deepEqual(collectModuleUsage(doc, excluded.map(id => ({id, title: id}))), []);
});

test('sorts results without changing the document or module catalog', () => {
  const doc = {
    _id: 'actor', name: '角色', flags: {zebra: {value: 1}, alpha: {value: 2}},
    items: [{_id: 'item', name: '条目', flags: {zebra: {value: 3}}}],
  };
  const modules = [{id: 'zebra', title: 'Z'}, {id: 'alpha', title: 'A'}];
  const original = structuredClone({doc, modules});
  const result = collectModuleUsage(doc, modules);
  assert.deepEqual(result.map(entry => entry.id), ['alpha', 'zebra']);
  assert.deepEqual(collectModuleUsage(doc, [...modules].reverse()), result);
  result[1].evidence[0].documentName = 'changed';
  assert.deepEqual({doc, modules}, original);
});

test('skips invalid legacy rows while keeping evidence from valid embedded entries', () => {
  const doc = {
    _id: 'actor', name: '角色', system: {rules: {}}, effects: {},
    items: [null, {_id: 'broken', system: {rules: 'invalid'}, effects: null}, {
      _id: 'item', name: '有效条目', items: {},
      system: {rules: [
        null,
        {key: 'Aura', effects: {}},
        {key: 'Aura', effects: [null, {uuid: 'Compendium.example.effects.Item.effect'}]},
        {key: 'ChoiceSet', choices: [null, {value: 'Compendium.example.items.Item.choice'}]},
        {key: 'GrantItem', uuid: 'Compendium.example.items.Item.feat'},
      ]},
    }],
  };
  assert.deepEqual(collectModuleUsage(doc), [{
    id: 'example', title: 'example', evidence: [
      {kind: 'reference', documentId: 'item', documentName: '有效条目', path: 'items.2.system.rules.2.effects.1.uuid'},
      {kind: 'reference', documentId: 'item', documentName: '有效条目', path: 'items.2.system.rules.4.uuid'},
    ],
  }]);
});

for (const type of ['ancestry', 'background', 'class']) {
  test(`collects the direct ${type} grant UUIDs`, () => {
    const doc = {
      _id: 'item', name: '构筑条目', type,
      system: {items: {
        grant: {uuid: 'Compendium.example.feats.Item.feat', level: 1},
        invalid: null,
        label: {uuid: 'handwritten-slug'},
      }},
    };
    assert.deepEqual(collectModuleUsage(doc), [{
      id: 'example', title: 'example', evidence: [
        {kind: 'reference', documentId: 'item', documentName: '构筑条目', path: 'system.items.grant.uuid'},
      ],
    }]);
  });
}

test('does not treat system.items on other item types as ABC grants', () => {
  for (const type of ['heritage', 'weapon']) {
    assert.deepEqual(collectModuleUsage({
      _id: 'item', name: '条目', type,
      system: {items: {entry: {uuid: 'Compendium.example.feats.Item.feat'}}},
    }), []);
  }
});
