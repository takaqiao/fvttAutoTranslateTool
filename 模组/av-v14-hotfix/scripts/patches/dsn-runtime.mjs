export function dsnCompatibility(g){
  const module=g.game?.modules?.get('dice-so-nice');
  if(!module?.active)return 'inactive-dsn';
  return null;
}
