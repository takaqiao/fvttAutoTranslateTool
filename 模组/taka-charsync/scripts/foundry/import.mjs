// 整卡导入：用一份完整文档覆盖卡（Foundry「导入数据」的做法），过不了校验的内嵌条目先挑出来，不让整次导入失败。
import { LOCAL_ACTOR_KEYS, LOCAL_ITEM_KEYS, OWN_FLAG } from '../lib/rules.mjs';

const EMBEDDED = [{name: 'items', documentName: 'Item'}, {name: 'effects', documentName: 'ActiveEffect'}];

const DRY_RUN_ROUNDS = 5;

/** 经当前世界的数据模型清洗一遍。不抛错，过不了校验的内嵌条目原样留在结果里。 */
export function cleanDoc(src) {
  return new Actor.implementation(foundry.utils.deepClone(src)).toObject();
}

// 条目的 ownership 里可能带着来源世界的用户 id（主服 GM 新建的条目），建卡服新建它时服务器去掉了这个键；
// 非 GM 再把它写回已有条目，服务器会以「只有 GM 能改 ownership」拒绝整次更新
function withLocalKeys(src, current) {
  if (!current) return src;
  const entry = {...src};
  for (const key of LOCAL_ITEM_KEYS) {
    if (current[key] === undefined) delete entry[key];
    else entry[key] = foundry.utils.deepClone(current[key]);
  }
  return entry;
}

const failureMessage = failure => (failure instanceof Error ? failure.message : String(failure));

// 预演失败时 `fields.<集合>.elements` 里的失败条目：整体预校验按数组下标记，逐条创建、更新时按 `_id` 记。
function failedEntries(error, payload) {
  const fields = error.getFailure()?.fields ?? {};
  const failed = [];
  for (const {name} of EMBEDDED) {
    for (const element of fields[name]?.elements ?? []) {
      const list = payload[name] ?? [];
      const src = typeof element.id === 'number' ? list[element.id] : list.find(entry => entry._id === element.id);
      if (src) failed.push({name, src, error: failureMessage(element.failure)});
    }
  }
  return failed;
}

/**
 * 用 `doc` 整卡覆盖 `actor`：`_id`、`ownership`、`folder`、`sort`、`_stats` 保留卡上的，`flags['taka-charsync']`
 * 换成 `flags`（调用方给写入后的完整内容），其余整份取 `doc`。内嵌条目按 id 增删改：卡上已有的条目
 * `_stats`、`ownership`、`sort` 也保留卡上的，新增的整份照抄 `doc`。
 *
 * 卡上已有的非法条目一律删掉（记进 `removed`）：留着会让整次更新在客户端预演时失败。
 * 传入的每个条目都先单独严格校验：条目自身的校验失败在 Foundry 的预演和更新里都不报错，那一条的改动被悄悄丢掉
 * （新增的没加上，已有的还是旧内容）。过不了的记进 `skipped`：新增的不加，已有的明确换回卡上现在的版本。
 * Foundry 在客户端校验失败时只弹通知、不抛错，`update` 返回空，这里当失败抛出。
 *
 * @returns {Promise<{skipped: {id: string, name: string, error: string}[], removed: {id: string, name: string}[]}>}
 */
export async function importFull(actor, doc, {flags}) {
  const payload = foundry.utils.deepClone(doc);
  for (const key of LOCAL_ACTOR_KEYS) delete payload[key];
  payload.flags = {...payload.flags, [OWN_FLAG]: flags};

  const removed = [];
  const skipped = [];
  // 整卡覆盖下从 payload 去掉已有条目就等于删掉它，所以已有的换回卡上现在的版本
  const skip = (name, src, error) => {
    skipped.push({id: src._id, name: src.name ?? '', error});
    const current = actor[name].has(src._id) && actor._source[name].find(entry => entry._id === src._id);
    return current ? [foundry.utils.deepClone(current)] : [];
  };

  for (const {name, documentName} of EMBEDDED) {
    if (!Array.isArray(payload[name])) continue;
    const collection = actor[name];
    // Foundry 重新初始化时不清 invalidDocumentIds，删掉或修好的条目还留在里面，以 _source 为准
    const invalid = new Set();
    for (const src of actor._source[name]) {
      if (collection.invalidDocumentIds.has(src._id) && !collection.has(src._id)) {
        invalid.add(src._id);
        removed.push({id: src._id, name: src.name ?? ''});
      }
    }
    const cls = CONFIG[documentName].documentClass;
    const current = new Map(actor._source[name].map(entry => [entry._id, entry]));
    payload[name] = payload[name].flatMap(src => {
      if (invalid.has(src._id)) return [];
      const entry = withLocalKeys(src, current.get(src._id));
      try {
        new cls(foundry.utils.deepClone(entry), {parent: actor, strict: true});
        return [entry];
      } catch (error) {
        return skip(name, entry, error.message);
      }
    });
  }

  for (let round = 1; ; round++) {
    try {
      actor.updateSource(foundry.utils.deepClone(payload), {dryRun: true, fallback: false, recursive: false});
      break;
    } catch (error) {
      const failed = error instanceof foundry.data.validation.DataModelValidationError ? failedEntries(error, payload) : [];
      if (!failed.length || round === DRY_RUN_ROUNDS) {
        throw new Error(`taka-charsync：${actor.name} 的整卡导入没有通过预演：${error.message}`, {cause: error});
      }
      for (const {name, src, error: message} of failed) {
        payload[name] = payload[name].flatMap(entry => (entry === src ? skip(name, src, message) : [entry]));
      }
    }
  }

  const updated = await actor.update(payload, {diff: false, recursive: false, takaCharsync: true});
  if (!updated) throw new Error(`taka-charsync：${actor.name} 的整卡导入被 Foundry 丢弃（客户端校验没有通过）`);
  return {skipped, removed};
}
