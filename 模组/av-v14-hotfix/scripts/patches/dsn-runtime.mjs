export function dsnCompatibility(g){
  if((g.game?.release?.generation??Number.parseInt(g.game?.version,10))!==14)return 'unsupported-core';
  const module=g.game?.modules?.get('dice-so-nice');
  if(!module?.active)return 'inactive-dsn';
  return Number.parseInt(module.version,10)===6?null:'unsupported-dsn';
}
