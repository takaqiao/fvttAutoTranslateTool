// 起步基线：上线前两边已经手工同步过的卡，认出同步时的那个版本，作为第一次合并的基线。
import { contentHash } from '../scripts/lib/canon.mjs';

/** 文档本身、条目、条目 effect 和 actor effect 的 `_stats.modifiedTime` 中最大的；一个都没有时为 0。 */
function lastModified(doc) {
  const items = doc.items ?? [];
  const embedded = [...items, ...items.flatMap(item => item.effects ?? []), ...(doc.effects ?? [])];
  return Math.max(0, ...[doc, ...embedded].map(entry => entry?._stats?.modifiedTime).filter(Number.isFinite));
}

/**
 * 要算起步基线的卡：每张种子卡和它的同伴，按卡 id 给出世界；`forced` 收 `bootstrap: 'main'` 的卡，同伴跟主人一样。
 * 同伴是种子里列出的，加上映射表 `cards` 里 `companionOf` 指向种子卡的（含按 `system.master.id` 认出的魔宠），
 * 后者的世界取映射里的。还没算出映射时 `cards` 省略。
 */
export function bootstrapTargets(seed, cards = {}) {
  const worlds = {};
  const forced = {};
  const add = (id, world, seedCard) => {
    worlds[id] = world;
    if (seedCard.bootstrap === 'main') forced[id] = 'main';
  };
  const seedCards = new Map(seed.cards.map(card => [card.charId, card]));
  for (const card of seed.cards) {
    for (const id of [card.charId, ...(card.companions ?? [])]) add(id, card.world, card);
  }
  for (const [id, entry] of Object.entries(cards)) {
    const master = seedCards.get(entry.companionOf);
    if (master) add(id, entry.world, master);
  }
  return {worlds, forced};
}

/**
 * 每张卡的起步基线。`cards` 是卡 id 列表；`mainDocs[cardId]` 是 `{doc, hash}`（`hash = contentHash(doc)`），
 * `charDocs[cardId]` 是建卡服文档，没有的卡不放；`since` 是两边手工同步完成的时刻（毫秒）。
 *
 * 「动过」指文档本身和全部内嵌条目 `_stats.modifiedTime` 的最大值晚于 `since`。
 * - `forced-main`：基线取主服当前版本，建卡服上的改动全部算草稿。
 * - `main-unchanged`：主服之后没动过，主服当前版本就是同步时的版本。
 * - `char-unchanged`：主服动过、建卡服没动过，建卡服文档就是同步时的主服版本。以它的内容哈希作主服版本引用
 *   （那时两边都还没有本模组的记号，不剥 flag），`historyDoc` 是它，由调用方发布成 `history/main-<hash>.json`。
 * - `both-changed`：两边都动过，认不出同步时的版本，基线留空，交给服主定。
 * - `missing`：没有主服文档；或者不是 `forced`、又没有建卡服文档。
 */
export async function bootstrapBases({cards, mainDocs, charDocs, since, forced = {}}) {
  const bases = {};
  for (const cardId of cards) {
    const main = mainDocs[cardId];
    const charDoc = charDocs[cardId];
    if (!main) {
      bases[cardId] = {case: 'missing', ref: null};
    } else if (forced[cardId] === 'main') {
      bases[cardId] = {case: 'forced-main', ref: {kind: 'main', hash: main.hash}};
    } else if (!charDoc) {
      bases[cardId] = {case: 'missing', ref: null};
    } else if (lastModified(main.doc) <= since) {
      bases[cardId] = {case: 'main-unchanged', ref: {kind: 'main', hash: main.hash}};
    } else if (lastModified(charDoc) <= since) {
      bases[cardId] = {case: 'char-unchanged', ref: {kind: 'main', hash: await contentHash(charDoc)}, historyDoc: charDoc};
    } else {
      bases[cardId] = {case: 'both-changed', ref: null};
    }
  }
  return bases;
}
