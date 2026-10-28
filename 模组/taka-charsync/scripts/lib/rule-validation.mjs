// 发布、预览和套用共用同一规则版本检查；不改写旧副本的历史记号。
export function copyRuleError(copy, snapshot) {
  if (snapshot?.version !== 2 || !Number.isSafeInteger(snapshot.enforcedSince) || snapshot.enforcedSince < 0) {
    return '读不到有效的团规则，请等待同步引擎重新发布';
  }
  const world = copy?.targetWorld;
  const hash = snapshot.hashes?.[world];
  if (!world || !Object.hasOwn(snapshot.rules ?? {}, world) || !/^[a-f0-9]{64}$/.test(hash ?? '')) {
    return '提交目标团的规则不可用，请等待同步引擎重新发布';
  }
  const mappedWorld = snapshot.cardWorld?.[copy.cardId];
  if (mappedWorld && mappedWorld !== world) return '工作卡所属团与副本提交目标不同，请核对角色归属';
  if (!Number.isSafeInteger(copy.rev) || copy.rev < 0) return '副本提交时间无效，不能核对采用的团规则';
  if (!copy.rulesHash && copy.rev < snapshot.enforcedSince) return null;
  if (copy.rulesHash !== hash) return '这份副本未采用当前团规则，请在工作卡中重新核对后提交';
  return null;
}
