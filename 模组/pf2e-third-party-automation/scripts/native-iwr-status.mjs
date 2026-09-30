const worlds=new Set(['-','sog','pnvfcgjbf2cjp7gz','team-automation-qa2']);
const values=collection=>Array.from(collection?.values?.()??collection??[]);
const hasEldamon=game=>values(game.actors).some(actor=>values(actor.items).some(item=>(item.sourceId??item._stats?.compendiumSource??item.flags?.core?.sourceId)==='Compendium.battlezoo-eldamon-pf2e.eldamon-features.Item.naawsnBug9EOpzfN'));

/** A setup-time system update can remove the bridge without restarting Node. */
export function notifyNativeIWRStatus({game,diagnostic,warn}){
 if(diagnostic.ready||!game.user?.isGM||game.user.id!==game.users?.activeGM?.id)return false;
 const eldamon=hasEldamon(game);if(!worlds.has(game.world?.id)&&!eldamon)return false;
 const detail=['missing-native-iwr-bridge','bridge-not-installed','native-bridge-unavailable'].includes(diagnostic.reason)
  ?'当前系统缺少已验证的 IWR 桥。系统更新后需由服务器管理员完成桥校验与服务重启，再刷新客户端。'
  :'当前系统未通过对应版本的 IWR 桥验证，请服务器管理员核对系统版本、文件和桥安装记录。';
 const unavailable=eldamon?'电击实伤追踪（蓄电／受电清理、反应电链）及依赖此桥的靖涛定风剑自动反应':'靖涛定风剑自动反应';
 warn(`${unavailable}未启用：${detail}普通伤害仍照常结算。`);
 return true;
}
