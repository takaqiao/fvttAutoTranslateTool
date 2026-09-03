import { createRequire } from 'node:module';
const require = createRequire('C:/Users/Taka/Desktop/fvtt/package.json');
const { ClassicLevel } = require('classic-level');
const db = new ClassicLevel(process.argv[2], { valueEncoding: 'json' });
await db.open();
for await (const [k, v] of db.iterator()) {
  if (!k.startsWith('!adventures!')) continue;
  console.log('=== journals (' + v.journal.length + ') ===');
  for (const j of v.journal) console.log(`  ${j.name}  [pages ${j.pages.length}]`);
  console.log('=== scenes (' + v.scenes.length + ') ===');
  for (const s of v.scenes) console.log('  ' + s.name);
  console.log('=== folders (' + v.folders.length + ') first 20 ===');
  for (const f of v.folders.slice(0,20)) console.log(`  ${f.name}  (type=${f.type})`);
  const ft = {}; for (const f of v.folders) ft[f.type] = (ft[f.type]||0)+1;
  console.log('  folder types:', JSON.stringify(ft));
  console.log('=== macros (' + v.macros.length + ') ===');
  for (const m of v.macros.slice(0,6)) console.log('  ' + m.name);
  console.log('=== tables (' + v.tables.length + ') ===');
  for (const t of v.tables) console.log(`  ${t.name} [results ${t.results?.length}]`);
  console.log('=== items (' + v.items.length + ') first 6 ===');
  for (const i of v.items.slice(0,6)) console.log('  ' + i.name);
  console.log('=== playlists (' + v.playlists.length + ') first 4 ===');
  for (const p of v.playlists.slice(0,4)) console.log(`  ${p.name} [sounds ${p.sounds?.length}]`);
}
await db.close();
