export function dsnCompatibility(g){
  const module=g.game?.modules?.get('dice-so-nice');
  if(!module?.active)return 'inactive-dsn';
  return Number.parseInt(module.version,10)===6?null:'unsupported-dsn';
}
