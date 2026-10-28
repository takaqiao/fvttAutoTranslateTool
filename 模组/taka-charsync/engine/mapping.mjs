// 映射表：建卡服卡 id → 它属于哪个团世界、对应主服哪张卡、卡主是谁、带哪些同伴；以及两边账号的对照。
import { OWN_FLAG } from '../scripts/lib/rules.mjs';
import { isFoundryId } from './config.mjs';

// Foundry 的 `User#isGM` 是 role >= ASSISTANT（3），助理 GM 也当 GM
const ASSISTANT_ROLE = 3;
const OWNER_LEVEL = 3;

const ownFlags = actor => actor?.flags?.[OWN_FLAG] ?? {};

/** `ownership` 里第一个权限为所有者、且是现有非 GM 用户的用户 id；卡不在或找不到时为 null。 */
export function charOwnerOf(actor, charUsers) {
  for (const [userId, level] of Object.entries(actor?.ownership ?? {})) {
    const user = charUsers.get(userId);
    if (level === OWNER_LEVEL && user && user.role < ASSISTANT_ROLE) return userId;
  }
  return null;
}

function buildCards({seed, charActors, charUsers, mainActors}) {
  const retired = (world, mainId) => Boolean(ownFlags(mainActors[world]?.get(mainId)).retired);

  // 先种子，再用主服上带 charId 的新卡覆盖；没读到的世界查不了 retired，种子照收
  const primaries = new Map();
  const seedCompanions = new Map();
  for (const {charId, world, mainId, companions = []} of seed.cards ?? []) {
    seedCompanions.set(charId, companions);
    if (!retired(world, mainId)) primaries.set(charId, {world, mainId});
  }
  for (const [world, actors] of Object.entries(mainActors)) {
    for (const actor of actors.values()) {
      const {charId, retired: isRetired} = ownFlags(actor);
      // 团世界里玩家能改自己卡的 flag，卡 id 又要拼进发布路径，不是 Foundry id 的不收
      if (isFoundryId(charId) && !isRetired) primaries.set(charId, {world, mainId: actor._id});
    }
  }

  const familiarsOf = new Map();
  for (const actor of charActors.values()) {
    const masterId = actor.type === 'familiar' ? actor.system?.master?.id : null;
    if (masterId) familiarsOf.set(masterId, [...(familiarsOf.get(masterId) ?? []), actor._id]);
  }

  const entry = (cardId, world, mainId, companions, companionOf) => ({
    world, mainId, charOwner: charOwnerOf(charActors.get(cardId), charUsers), companions, companionOf,
  });
  const cards = {};
  const companionEntries = [];
  for (const [charId, {world, mainId}] of primaries) {
    // 魔宠要主服同一世界里有同 id 的 actor 才算：建卡服的草稿备份卡也指着同一个主人
    const familiars = (familiarsOf.get(charId) ?? []).filter(id => mainActors[world]?.has(id));
    const companions = [...new Set([...(seedCompanions.get(charId) ?? []), ...familiars])].filter(id => !retired(world, id));
    cards[charId] = entry(charId, world, mainId, companions, null);
    for (const id of companions) companionEntries.push([id, entry(id, world, id, [], charId)]);
  }
  // 同伴条目后写，同一个 id 既是主卡又是同伴时按同伴算
  for (const [id, companion] of companionEntries) cards[id] = companion;
  return cards;
}

/** 名字 → 用户 id，只收恰好一个用户用这个名字的；`skip` 里的用户不参与。 */
function uniqueIdsByName(users, skip) {
  const byName = new Map();
  for (const [id, {name}] of users) {
    if (!skip.has(id)) byName.set(name, byName.has(name) ? null : id);
  }
  return byName;
}

function pairUsers({seed, charUsers, worldUsers}) {
  const users = {};
  const link = (charUser, world, worldUser) => {
    (users[charUser] ??= {})[world] = worldUser;
  };
  const seeded = seed.users ?? [];
  for (const {world, worldUser, charUser} of seeded) link(charUser, world, worldUser);

  // 种子里出现过的账号在那个世界不再按名字配；重名的不配，免得配错人
  for (const [world, usersOfWorld] of Object.entries(worldUsers)) {
    const seededHere = seeded.filter(pair => pair.world === world);
    const charByName = uniqueIdsByName(charUsers, new Set(seededHere.map(pair => pair.charUser)));
    const worldByName = uniqueIdsByName(usersOfWorld, new Set(seededHere.map(pair => pair.worldUser)));
    for (const [name, worldUser] of worldByName) {
      const charUser = charByName.get(name);
      if (worldUser && charUser) link(charUser, world, worldUser);
    }
  }
  return users;
}

/**
 * 生成 `meta.json` 的 `cards` 与 `users`。
 *
 * `seed.cards` 每项 `{charId, world, mainId, companions?}`，`companions` 是同伴在建卡服的 id（主服同 id）；
 * `seed.users` 每项 `{world, worldUser, charUser}`。`charActors`、`mainActors[w]` 是 actor 行（`scanActors` 的结果），
 * `charUsers`、`worldUsers[w]` 是 `readUsers` 的结果。本轮没读到的世界不在 `mainActors` / `worldUsers` 里。
 *
 * 主服卡带 `retired` 的不进映射；同伴各占一条，`companionOf` 是主人的 cardId。
 */
export function buildMapping({seed, charActors, charUsers, mainActors, worldUsers}) {
  return {
    cards: buildCards({seed, charActors, charUsers, mainActors}),
    users: pairUsers({seed, charUsers, worldUsers}),
  };
}
