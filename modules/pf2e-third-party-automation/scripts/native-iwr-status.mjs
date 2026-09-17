const worlds=new Set(['-','sog','pnvfcgjbf2cjp7gz','team-automation-qa2']);

/** A setup-time system update can remove the bridge without restarting Node. */
export function notifyNativeIWRStatus({game,diagnostic,warn}){
 if(diagnostic.ready||!worlds.has(game.world?.id)||!game.user?.isGM||game.user.id!==game.users?.activeGM?.id)return false;
 const detail=['missing-native-iwr-bridge','bridge-not-installed','native-bridge-unavailable'].includes(diagnostic.reason)
  ?'当前系统缺少已验证的 IWR 桥。系统更新后需由服务器管理员完成桥校验与服务重启，再刷新客户端。'
  :'当前系统未通过对应版本的 IWR 桥验证，请服务器管理员核对系统版本、文件和桥安装记录。';
 warn(`靖涛定风剑自动反应未启用：${detail}普通伤害仍照常结算。`);
 return true;
}
