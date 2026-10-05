import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { inflateRawSync } from 'node:zlib';
import { pathToFileURL } from 'node:url';
import { runtimeFiles } from './build-delivery.mjs';

const expectedFiles = new Set([...runtimeFiles.map(f => `module/${f}`), 'obs/FVTT-OBS-Director.scene-collection.json', 'obs/profile.ini', 'macros/sync-recorder.js', 'defaults.json', 'settings-template.json', '开始使用.md', '来源与验收.md', 'manifest.json']);
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const crcTable = Array.from({ length: 256 }, (_, i) => { let n = i; for (let bit = 0; bit < 8; bit++) n = (n & 1) ? (0xedb88320 ^ (n >>> 1)) : n >>> 1; return n >>> 0; });
export function crc32(bytes) { let crc = 0xffffffff; for (const byte of bytes) crc = crcTable[(crc ^ byte) & 255] ^ (crc >>> 8); return (crc ^ 0xffffffff) >>> 0; }
function allowed(name) { if (!expectedFiles.has(name)) throw Error('Delivery path outside allowlist'); }
function scan(name, bytes, secrets = []) {
  if (!/\.(json|mjs|js|css|ini|md|svg)$/.test(name)) return;
  const text = bytes.toString('utf8');
  if (secrets.some(secret => text.includes(secret))) throw Error('Delivery private authentication value scan refused');
  if (/[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}/.test(text) || /(?:password|server_password|authToken|apiKey|secret)\s*["']?\s*[:=]\s*["']?[^\s"',;}]+/i.test(text)) throw Error('Delivery secret scan refused');
  if (/C:[\\/]+Users[\\/]|\/root\/|\.codex[\\/]|obs-director-qa-|qa-credentials|native-inputs|qa-helper|127\.0\.0\.[1-7](?::\d+)?/i.test(text)) throw Error('Delivery private machine path scan refused');
  if (/\.json$/.test(name)) JSON.parse(text);
}
function checkEntries(entries, secrets = []) {
  if (entries.size !== expectedFiles.size || [...expectedFiles].some(name => !entries.has(name))) throw Error('Delivery allowlist is incomplete');
  for (const [name, bytes] of entries) { allowed(name); scan(name, bytes, secrets); }
  const manifest = JSON.parse(entries.get('manifest.json'));
  if (manifest.schemaVersion !== 1 || !Array.isArray(manifest.files) || manifest.files.length !== entries.size - 1) throw Error('Invalid packaging manifest');
  const seen = new Set();
  for (const item of manifest.files) {
    allowed(item.path);
    if (item.path === 'manifest.json' || seen.has(item.path) || !entries.has(item.path)) throw Error('Ambiguous packaging manifest'); seen.add(item.path);
    const bytes = entries.get(item.path); if (item.bytes !== bytes.length || item.sha256 !== hash(bytes)) throw Error('Delivery SHA verification failed');
  }
  const module = JSON.parse(entries.get('module/module.json'));
  if (module.id !== 'taka-obs-director') throw Error('Delivery module identity refused');
  for (const file of [...module.esmodules ?? [], ...module.styles ?? []]) if (!entries.has(`module/${file}`)) throw Error('Missing relative module reference');
  for (const [name, bytes] of entries) {
    if (name.endsWith('.mjs') || name.endsWith('.js')) {
      for (const match of bytes.toString().matchAll(/(?:from\s*|import\s*\()\s*['"]([^'"]+)['"]/g)) {
        const relative = match[1]; if (!relative.startsWith('.')) throw Error('Unexpected delivery dependency');
        const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(name), relative));
        if (!resolved.startsWith('module/') || !entries.has(resolved)) throw Error('Missing relative script reference');
      }
    }
  }
  // Every approved skin asset is resolved relative to the installed module.
  for (const skin of ['cotct', 'sog', 'fotrp', 'av', 'bob']) for (const asset of [`frames/${skin}-frame.png`, `logos/${skin}.webp`]) if (!entries.has(`module/assets/${asset}`)) throw Error('Missing relative skin asset');
  const preset = JSON.parse(entries.get('obs/FVTT-OBS-Director.scene-collection.json'));
  const browser = preset.sources?.filter(source => source.id === 'browser_source');
  if (browser?.length !== 1 || browser[0].settings.url !== 'https://v14.taka.wang/game' || browser[0].settings.reroute_audio !== true || browser[0].mixers !== 1 || browser[0].monitoring_type !== 0) throw Error('Portable OBS preset refused');
  return { files: entries.size, shaVerified: true, allowlistVerified: true, secretsScanPassed: true, relativeReferencesVerified: true };
}
function directoryEntries(root) {
  const entries = new Map();
  function visit(directory, prefix = '') {
    for (const name of fs.readdirSync(directory).sort()) {
      const full = path.join(directory, name), relative = prefix + name, stat = fs.lstatSync(full);
      if (stat.isSymbolicLink()) throw Error('Delivery symlink refused');
      if (stat.isDirectory()) visit(full, relative + '/');
      else if (stat.isFile()) { allowed(relative); entries.set(relative, fs.readFileSync(full)); }
      else throw Error('Delivery non-file refused');
    }
  }
  visit(root); return entries;
}
function secretValues(privateConfigs) {
  const values = [];
  function visit(value) {
    if (!value || typeof value !== 'object') return;
    for (const [key, child] of Object.entries(value)) {
      if (/^(?:password|adminPassword|server_password|controlToken|apiKey|authToken|room)$/i.test(key) && typeof child === 'string' && child.length >= 8) values.push(child);
      else if (typeof child === 'object') visit(child);
    }
  }
  for (const file of privateConfigs) visit(JSON.parse(fs.readFileSync(file)));
  return [...new Set(values)];
}
export function verifyDirectory(root, { privateConfigs = [] } = {}) { return checkEntries(directoryEntries(root), secretValues(privateConfigs)); }

// A standard UTF-8, stored ZIP keeps packaging dependency-free and reviewable.
function zipBytes(entries) {
  const local = [], central = []; let offset = 0;
  for (const [name, bytes] of entries) {
    const encoded = Buffer.from(name), crc = crc32(bytes), header = Buffer.alloc(30);
    header.writeUInt32LE(0x04034b50); header.writeUInt16LE(20, 4); header.writeUInt16LE(0x800, 6); header.writeUInt16LE((46 << 9) | 33, 12);
    header.writeUInt32LE(crc, 14); header.writeUInt32LE(bytes.length, 18); header.writeUInt32LE(bytes.length, 22); header.writeUInt16LE(encoded.length, 26);
    local.push(header, encoded, bytes);
    const directory = Buffer.alloc(46); directory.writeUInt32LE(0x02014b50); directory.writeUInt16LE(0x314, 4); directory.writeUInt16LE(20, 6); directory.writeUInt16LE(0x800, 8); directory.writeUInt16LE((46 << 9) | 33, 14);
    directory.writeUInt32LE(crc, 16); directory.writeUInt32LE(bytes.length, 20); directory.writeUInt32LE(bytes.length, 24); directory.writeUInt16LE(encoded.length, 28); directory.writeUInt32LE((0o100644 << 16) >>> 0, 38); directory.writeUInt32LE(offset, 42);
    central.push(directory, encoded); offset += header.length + encoded.length + bytes.length;
  }
  const directory = Buffer.concat(central), end = Buffer.alloc(22); end.writeUInt32LE(0x06054b50); end.writeUInt16LE(entries.size, 8); end.writeUInt16LE(entries.size, 10); end.writeUInt32LE(directory.length, 12); end.writeUInt32LE(offset, 16);
  return Buffer.concat([...local, directory, end]);
}
export function verifyZip(bytes) {
  if (bytes.length < 22 || bytes.length > 64 * 1024 * 1024 || bytes.readUInt32LE(bytes.length - 22) !== 0x06054b50) throw Error('Invalid ZIP header');
  const end = bytes.length - 22, count = bytes.readUInt16LE(end + 10), size = bytes.readUInt32LE(end + 12), start = bytes.readUInt32LE(end + 16);
  if (bytes.readUInt32LE(end + 4) !== 0 || bytes.readUInt16LE(end + 8) !== count || bytes.readUInt16LE(end + 20) !== 0 || start + size !== end || count !== expectedFiles.size) throw Error('Unsupported ZIP header');
  const entries = new Map(); let cursor = start, localEnd = 0;
  for (let i = 0; i < count; i++) {
    if (cursor + 46 > end || bytes.readUInt32LE(cursor) !== 0x02014b50) throw Error('Invalid ZIP central header');
    const flags = bytes.readUInt16LE(cursor + 8), method = bytes.readUInt16LE(cursor + 10), crc = bytes.readUInt32LE(cursor + 16), compressed = bytes.readUInt32LE(cursor + 20), length = bytes.readUInt32LE(cursor + 24), nameLength = bytes.readUInt16LE(cursor + 28), extra = bytes.readUInt16LE(cursor + 30), comment = bytes.readUInt16LE(cursor + 32), offset = bytes.readUInt32LE(cursor + 42);
    if (flags !== 0x800 || ![0, 8].includes(method) || extra || comment || length > 32 * 1024 * 1024 || offset !== localEnd || bytes.readUInt16LE(cursor + 34) !== 0 || (bytes.readUInt32LE(cursor + 38) >>> 16 & 0o170000) !== 0o100000) throw Error('Unsupported ZIP entry header');
    const nameBytes = bytes.subarray(cursor + 46, cursor + 46 + nameLength), name = nameBytes.toString('utf8'); allowed(name);
    if (entries.has(name) || offset + 30 + nameLength + compressed > start || bytes.readUInt32LE(offset) !== 0x04034b50 || bytes.readUInt16LE(offset + 6) !== flags || bytes.readUInt16LE(offset + 8) !== method || bytes.readUInt32LE(offset + 14) !== crc || bytes.readUInt32LE(offset + 18) !== compressed || bytes.readUInt32LE(offset + 22) !== length || bytes.readUInt16LE(offset + 26) !== nameLength || bytes.readUInt16LE(offset + 28) !== 0 || !nameBytes.equals(bytes.subarray(offset + 30, offset + 30 + nameLength))) throw Error('ZIP local header mismatch');
    const packed = bytes.subarray(offset + 30 + nameLength, offset + 30 + nameLength + compressed), decoded = method === 0 ? packed : inflateRawSync(packed, { maxOutputLength: 32 * 1024 * 1024 });
    if (decoded.length !== length || crc32(decoded) !== crc) throw Error('ZIP CRC verification failed');
    entries.set(name, decoded); localEnd = offset + 30 + nameLength + compressed; cursor += 46 + nameLength;
  }
  if (cursor !== end || localEnd !== start) throw Error('Unexpected ZIP trailing header/data');
  return { ...checkEntries(entries), crcVerified: true, entries };
}
export function packageDelivery({ directory, zip, extract, privateConfigs = [] }) {
  if (!zip || !extract || fs.existsSync(zip) || fs.existsSync(extract)) throw Error('New ZIP and second extraction paths required');
  const source = path.resolve(directory);
  for (const target of [zip, extract]) if (path.resolve(target) === source || path.resolve(target).startsWith(source + path.sep)) throw Error('Delivery outputs must be outside the source folder');
  const secrets = secretValues(privateConfigs), entries = directoryEntries(directory); checkEntries(entries, secrets);
  const archive = zipBytes(entries), verified = verifyZip(archive);
  fs.writeFileSync(zip, archive, { flag: 'wx' });
  fs.mkdirSync(extract, { recursive: true });
  for (const [name, bytes] of verified.entries) { const target = path.join(extract, ...name.split('/')); fs.mkdirSync(path.dirname(target), { recursive: true }); fs.writeFileSync(target, bytes, { flag: 'wx' }); }
  const moved = verifyDirectory(extract, { privateConfigs });
  return { ...moved, crcVerified: true, zipSha256: hash(fs.readFileSync(zip)), zipBytes: archive.length, secondExtractionVerified: true };
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    const a = {}, privateConfigs = []; for (let i = 2; i < process.argv.length; i += 2) { if (!process.argv[i].startsWith('--') || !process.argv[i + 1]) throw Error('Expected named arguments'); const key = process.argv[i].slice(2); if (key === 'private-config') privateConfigs.push(process.argv[i + 1]); else a[key] = process.argv[i + 1]; }
    const result = a.zip ? packageDelivery({ directory: a.directory, zip: a.zip, extract: a.extract, privateConfigs }) : verifyDirectory(a.directory, { privateConfigs });
    console.log(JSON.stringify(result));
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
