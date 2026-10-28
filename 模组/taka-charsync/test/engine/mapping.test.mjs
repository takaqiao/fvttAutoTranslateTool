import test from 'node:test';
import assert from 'node:assert/strict';
import { buildMapping } from '../../engine/mapping.mjs';

const GM = 'userGM0000000001';
const ASSISTANT = 'userAS0000000001';
const PLAYER_A = 'userPA0000000001';
const PLAYER_B = 'userPB0000000001';

const charActor = (id, overrides = {}) => ({_id: id, type: 'character', system: {}, ownership: {default: 0}, flags: {}, ...overrides});
const mainActor = (id, ownFlags = {}) => ({_id: id, type: 'character', system: {}, flags: {'taka-charsync': ownFlags}});
const byId = actors => new Map(actors.map(actor => [actor._id, actor]));
const usersOf = entries => new Map(entries.map(([id, name, role = 1]) => [id, {name, role, character: null}]));

function cardsFixture() {
  const seed = {
    cards: [
      {charId: 'cardHero00000001', world: 'w', mainId: 'cardHero00000001', companions: ['cardEido00000001']},
      {charId: 'cardOld000000001', world: 'w', mainId: 'mainOld000000001'},
    ],
    users: [],
  };
  const charActors = byId([
    // 第一个值为 3 的是 GM，第二个是助理 GM，都不算卡主
    charActor('cardHero00000001', {ownership: {default: 0, [GM]: 3, [ASSISTANT]: 3, [PLAYER_A]: 3}}),
    charActor('cardEido00000001', {ownership: {default: 0, [PLAYER_A]: 3}}),
    charActor('cardFami00000001', {type: 'familiar', system: {master: {id: 'cardHero00000001'}}, ownership: {[PLAYER_A]: 3}}),
    // 主服没有同 id 的 actor（比如草稿备份卡），不算同伴
    charActor('cardFami00000002', {type: 'familiar', system: {master: {id: 'cardHero00000001'}}, ownership: {[PLAYER_A]: 3}}),
    charActor('cardNew000000001', {ownership: {default: 2, [PLAYER_B]: 3}}),
  ]);
  const mainActors = {
    w: byId([
      mainActor('cardHero00000001'),
      mainActor('cardEido00000001'),
      mainActor('cardFami00000001'),
      mainActor('mainOld000000001', {retired: true}),
      mainActor('mainNew000000001', {charId: 'cardNew000000001'}),
      // 新卡后来又被替换掉
      mainActor('mainGone00000001', {charId: 'cardGone00000001', retired: true}),
    ]),
  };
  const charUsers = usersOf([[GM, 'Gamemaster', 4], [ASSISTANT, 'Assistant', 3], [PLAYER_A, 'Player A'], [PLAYER_B, 'Player B']]);
  return {seed, charActors, charUsers, mainActors, worldUsers: {w: new Map()}};
}

test('buildMapping combines seed cards and flagged new cards, drops retired ones, adds companions', () => {
  const {cards} = buildMapping(cardsFixture());
  assert.deepEqual(cards, {
    cardHero00000001: {
      world: 'w', mainId: 'cardHero00000001', charOwner: PLAYER_A,
      companions: ['cardEido00000001', 'cardFami00000001'], companionOf: null,
    },
    cardEido00000001: {world: 'w', mainId: 'cardEido00000001', charOwner: PLAYER_A, companions: [], companionOf: 'cardHero00000001'},
    cardFami00000001: {world: 'w', mainId: 'cardFami00000001', charOwner: PLAYER_A, companions: [], companionOf: 'cardHero00000001'},
    cardNew000000001: {world: 'w', mainId: 'mainNew000000001', charOwner: PLAYER_B, companions: [], companionOf: null},
  });
});

test('buildMapping keeps a flagged card whose seed mapping was retired', () => {
  const fixture = cardsFixture();
  fixture.mainActors.w.set('mainRepl00000001', mainActor('mainRepl00000001', {charId: 'cardOld000000001'}));
  const {cards} = buildMapping(fixture);
  assert.equal(cards.cardOld000000001.mainId, 'mainRepl00000001');
});

test('buildMapping ignores a charId flag that is not a Foundry id', () => {
  const fixture = cardsFixture();
  fixture.mainActors.w.set('mainEvil00000001', mainActor('mainEvil00000001', {charId: '../../evil'}));
  const {cards} = buildMapping(fixture);
  assert.deepEqual(Object.keys(cards).sort(), ['cardEido00000001', 'cardFami00000001', 'cardHero00000001', 'cardNew000000001']);
});

test('buildMapping keeps seed cards of worlds it could not read, with no owner when the char card is missing', () => {
  const fixture = cardsFixture();
  fixture.seed.cards.push({charId: 'cardElse00000001', world: 'v', mainId: 'cardElse00000001'});
  const {cards} = buildMapping(fixture);
  assert.deepEqual(cards.cardElse00000001, {world: 'v', mainId: 'cardElse00000001', charOwner: null, companions: [], companionOf: null});
});

test('buildMapping pairs users by exact name and lets the seed win', () => {
  const charUsers = usersOf([
    ['charPlayerA00001', 'Player A'],
    ['charPlayerB00001', 'Player B'],
    ['charPlayerD00001', 'Player D'],
    ['charPlayerD2Seed', 'Player D (alt)'],
    ['charPlayerE00001', 'Player E'],
    ['charPlayerE00002', 'Player E'],
    ['charPlayerAlow01', 'player a'],
  ]);
  const worldUsers = {
    w: usersOf([
      ['wPlayerA0000001', 'Player A'],
      ['wPlayerB0000001', 'Player B'],
      ['wPlayerB2000001', 'Player B2'],
      ['wPlayerD0000001', 'Player D'],
      ['wPlayerE0000001', 'Player E'],
    ]),
    v: usersOf([['vPlayerA0000001', 'Player A']]),
  };
  const seed = {
    cards: [],
    users: [
      // 按名字本该配 wPlayerB0000001，种子改配到另一个世界用户
      {world: 'w', worldUser: 'wPlayerB2000001', charUser: 'charPlayerB00001'},
      // 种子占用的世界用户不再按名字配给同名的建卡服账号
      {world: 'w', worldUser: 'wPlayerD0000001', charUser: 'charPlayerD2Seed'},
    ],
  };
  const {users} = buildMapping({seed, charActors: new Map(), charUsers, mainActors: {}, worldUsers});
  assert.deepEqual(users, {
    charPlayerA00001: {w: 'wPlayerA0000001', v: 'vPlayerA0000001'},
    charPlayerB00001: {w: 'wPlayerB2000001'},
    charPlayerD2Seed: {w: 'wPlayerD0000001'},
  });
});
