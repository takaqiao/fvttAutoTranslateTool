import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

export const runtimeFiles = [
  'module.json', 'main.mjs', 'scripts/model.mjs', 'scripts/pf2e.mjs', 'scripts/view.mjs',
  'scripts/skins.mjs', 'scripts/runtime.mjs', 'scripts/icons.mjs', 'scripts/client.mjs', 'scripts/viewport.mjs', 'scripts/portrait-layout.mjs', 'scripts/portrait-defaults.mjs', 'scripts/portrait-editor.mjs', 'styles/director.css',
  ...['cotct', 'sog', 'fotrp', 'av', 'bob'].map((skin) => `assets/frames/${skin}-frame.png`),
  ...['cotct', 'sog', 'fotrp', 'av', 'bob'].map((skin) => `assets/logos/${skin}.webp`),
  'assets/logos/cotct-title.svg', 'assets/textures/focus-material-atlas.png', 'assets/textures/cast-nameplate-atlas.png',
];
const extras = ['obs/FVTT-OBS-Director.scene-collection.json', 'obs/profile.ini', 'macros/sync-recorder.js', 'defaults.json', 'settings-template.json'];
export async function buildDelivery({ moduleRoot = fileURLToPath(new URL('..', import.meta.url)), destination, guideFiles = [] }) {
  if (!destination || fs.existsSync(destination)) throw Error('Delivery destination already exists or is missing');
  const selected = [...runtimeFiles.map((file) => ({ source: path.join(moduleRoot, file), path: `module/${file}` })), ...extras.map((file) => ({ source: path.join(moduleRoot, file), path: file }))];
  const allowedGuides = new Set(['开始使用.md', '来源与验收.md']);
  for (const file of guideFiles) {
    if (!allowedGuides.has(path.basename(file))) throw Error('Guide is not in the delivery whitelist');
    selected.push({ source: file, path: path.basename(file) });
  }
  for (const file of selected) if (!fs.statSync(file.source).isFile()) throw Error('Missing delivery file');
  fs.mkdirSync(destination, { recursive: true });
  const files = [];
  for (const file of selected) {
    const bytes = fs.readFileSync(file.source);
    const target = path.join(destination, file.path);
    fs.mkdirSync(path.dirname(target), { recursive: true }); fs.writeFileSync(target, bytes);
    files.push({ path: file.path, bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') });
  }
  const report = { schemaVersion: 1, files };
  fs.writeFileSync(path.join(destination, 'manifest.json'), JSON.stringify(report, null, 2));
  return report;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { args } = await import('./deploy.cjs').then((m) => m.default);
  try { const a = args(process.argv.slice(2)); console.log(JSON.stringify(await buildDelivery({ moduleRoot: a['module-root'], destination: a.destination, guideFiles: [a['start-guide'], a['evidence-guide']].filter(Boolean) }))); }
  catch (error) { console.error(error.message); process.exitCode = 1; }
}
