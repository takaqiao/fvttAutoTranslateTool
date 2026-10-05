import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { buildDelivery } from '../tools/build-delivery.mjs';
const file = new URL('../tools/verify-delivery.mjs', import.meta.url);

test('portable ZIP verifies CRC/SHA/allowlist and a second extraction with relative assets', async () => {
  assert.ok(fs.existsSync(file), 'Package verifier must exist');
  const { packageDelivery, verifyZip, verifyDirectory } = await import(file);
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'director-package-'));
  try {
    const guides = ['开始使用.md', '来源与验收.md'].map(name => { const f = path.join(root, name); fs.writeFileSync(f, '真实说明，无秘密。'); return f; });
    const delivery = path.join(root, 'delivery'); await buildDelivery({ destination: delivery, guideFiles: guides });
    const zip = path.join(root, 'portable.zip'), moved = path.join(root, 'different/location');
    const result = packageDelivery({ directory: delivery, zip, extract: moved });
    assert.equal(result.files, 35); assert.equal(result.crcVerified, true); assert.equal(result.shaVerified, true); assert.equal(result.relativeReferencesVerified, true);
    assert.equal((await verifyDirectory(moved)).files, 35);
    assert.deepEqual(fs.readFileSync(path.join(moved, 'module/scripts/portrait-layout.mjs')), fs.readFileSync(new URL('../scripts/portrait-layout.mjs', import.meta.url)));
    assert.ok(fs.existsSync(path.join(moved, 'module/assets/textures/cast-nameplate-atlas.png')));
    assert.equal(JSON.parse(fs.readFileSync(path.join(moved, 'obs/FVTT-OBS-Director.scene-collection.json'))).sources[0].settings.url, 'https://v14.taka.wang/game');
    const independent = JSON.parse(execFileSync('python', ['-c', 'import sys,zipfile,json,hashlib\nz=zipfile.ZipFile(sys.argv[1]); assert z.testzip() is None\nm=json.loads(z.read("manifest.json"))\nassert all(hashlib.sha256(z.read(f["path"])).hexdigest()==f["sha256"] for f in m["files"])\nprint(json.dumps({"files":len(z.namelist()),"crc":True,"sha":True}))', zip], { encoding: 'utf8', windowsHide: true }));
    assert.deepEqual(independent, { files: 35, crc: true, sha: true });
    const damaged = fs.readFileSync(zip); const nameLength = damaged.readUInt16LE(26); damaged[30 + nameLength] ^= 1;
    assert.throws(() => verifyZip(damaged), /CRC|header/);
    fs.writeFileSync(path.join(delivery, 'private-token.json'), '{}'); assert.throws(() => verifyDirectory(delivery), /allowlist/);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('matching manifest hashes cannot authorize credentials, JWTs or machine QA paths', async () => {
  const { verifyDirectory } = await import(file);
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'director-secret-scan-'));
  try {
    const guides = ['开始使用.md', '来源与验收.md'].map(name => { const f = path.join(root, name); fs.writeFileSync(f, '说明'); return f; });
    const delivery = path.join(root, 'delivery'); await buildDelivery({ destination: delivery, guideFiles: guides });
    const guide = path.join(delivery, '来源与验收.md'), manifestPath = path.join(delivery, 'manifest.json');
    for (const content of ['password="do-not-package"', 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhYnNkZWZnaGprbG1ub3BxciJ9.abcdefghijklmnopqrstuvwxyz123456', 'C:\\Users\\Taka\\private\\obs-director-qa-cotct']) {
      const bytes = Buffer.from(content); fs.writeFileSync(guide, bytes); const manifest = JSON.parse(fs.readFileSync(manifestPath));
      const entry = manifest.files.find(f => f.path === '来源与验收.md'); entry.sha256 = crypto.createHash('sha256').update(bytes).digest('hex'); entry.bytes = bytes.length; fs.writeFileSync(manifestPath, JSON.stringify(manifest));
      assert.throws(() => verifyDirectory(delivery), /secret|private|machine/i);
    }
    const secret = 'bare-high-entropy-private-value'; fs.writeFileSync(guide, secret);
    const manifest = JSON.parse(fs.readFileSync(manifestPath)), entry = manifest.files.find(f => f.path === '来源与验收.md'); entry.bytes = Buffer.byteLength(secret); entry.sha256 = crypto.createHash('sha256').update(secret).digest('hex'); fs.writeFileSync(manifestPath, JSON.stringify(manifest));
    const privateConfig = path.join(root, 'private.json'); fs.writeFileSync(privateConfig, JSON.stringify({ server_password: secret }));
    assert.throws(() => verifyDirectory(delivery, { privateConfigs: [privateConfig] }), /authentication value/);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});
