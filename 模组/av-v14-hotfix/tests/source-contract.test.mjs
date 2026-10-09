import test from 'node:test';
import assert from 'node:assert/strict';
import {existsSync} from 'node:fs';
import vm from 'node:vm';

const url = new URL('../scripts/source-contract.mjs', import.meta.url);
const contract = existsSync(url) ? await import(url) : {};
const matches = (source, declarations) => contract.matchesDeclarations?.(source, declarations) ?? false;
const selection = "function select(token) { return token.actor.items.filter(item => item.type === 'effect'); }";

test('local declarations accept formatting changes and unrelated code', () => {
  assert.equal(matches(`const unrelated = 2;\n${selection.replace('return', '/* explanation */ return')}\nfunction other() { return 9; }`, {select:selection}), true);
});

test('comment, string, template, regexp and nested declaration decoys cannot prove a source contract', () => {
  for (const decoy of [
    `/* ${selection} */`, `// ${selection}\n`, `const note = ${JSON.stringify(selection)};`,
    'const note = `' + selection + '`;', 'const note = `outer ${`' + selection + '`}`;',
    `const note = /${selection.replaceAll('/', '\\/')}/;`, `function outer() { ${selection} }`,
    `const expression = ${selection};`, `const expression = async ${selection};`, `const note = 1 + /${selection}/;`,
    `if (false) {} else /; ${selection}/;`, `let note = ''; note += /; ${selection}/;`, `debugger\n/; ${selection}/;`
  ]) assert.equal(matches(decoy, {select:selection}), false, decoy);
});

const regexpDecoy = `/; ${selection};x/`;
for (const [context, source] of [
  ['else', `if (false) {} else ${regexpDecoy};`],
  ['do', `do ${regexpDecoy}; while (false);`],
  ['instanceof', `({}) instanceof ${regexpDecoy};`],
  ['debugger newline', `debugger\n${regexpDecoy};`],
  ['in', `'name' in ${regexpDecoy};`],
  ['of', `for (const value of ${regexpDecoy}) {}`],
  ['yield delegation', `function* generator() { yield* ${regexpDecoy}; }`],
  ['compound assignment', `let value; value += ${regexpDecoy};`],
  ['exponent assignment', `let value; value **= ${regexpDecoy};`],
  ['logical assignment', `let value; value &&= ${regexpDecoy};`],
  ['division operand', `const value = 2 / ${regexpDecoy}.source.length;`],
  ['control statement', `if (false) ${regexpDecoy};`],
  ['uninitialized declaration', `let name\n${regexpDecoy};`],
  ['multiple declarations', `var first, second\n${regexpDecoy};`],
  ['break label', `outer: while (false) { break outer\n${regexpDecoy}; }`],
  ['continue label', `outer: while (false) { continue outer\n${regexpDecoy}; }`]
]) test(`regexp after ${context} cannot expose a declaration`, () => {
  // These are syntactically valid JavaScript, not malformed-source rejections.
  assert.doesNotThrow(() => new vm.Script(source));
  assert.equal(matches(source, {select:selection}), false);
  let tokens;
  try { tokens = contract.sourceTokens(source); } catch { return; }
  assert.equal(tokens.some(token => token.value === 'select'), false);
});

test('a regexp after an unterminated import declaration cannot expose source tokens', () => {
  const source = `import dependency from 'package'\n${regexpDecoy};`;
  assert.equal(matches(source, {select:selection}), false);
});

test('unambiguous division and regular expression operands remain readable', () => {
  const source = `${selection}\nconst infinity = 1 / 0; const half = count / 2; const match = /value/.test(text);`;
  assert.equal(matches(source, {select:selection}), true);
});

test('every JavaScript line terminator ends a line comment', () => {
  for (const newline of ['\r', '\u2028', '\u2029']) {
    const source = `${selection}\n// comment${newline}select = foreign;`;
    assert.doesNotThrow(() => new vm.Script(source));
    assert.equal(matches(source, {select:selection}), false);
  }
});

test('legacy HTML comment text cannot supply a source declaration', () => {
  for (const marker of ['<!--', '-->']) {
    const source = `${marker} ; ${selection};x\n`;
    assert.doesNotThrow(() => new vm.Script(source));
    assert.equal(matches(source, {select:selection}), false);
  }
});

test('changed dependencies and duplicate or reassigned bindings fail a contract', () => {
  for (const source of [selection.replace("'effect'", "'spell'"), `async ${selection}`, `${selection}\n${selection}`, `${selection}\nselect = foreign;`]) {
    assert.equal(matches(source, {select:selection}), false);
  }
});

test('property assignments and Object prototype names are not local binding replacements', () => {
  assert.equal(matches(`${selection}\napi.select = select; api.toString(); api.constructor();`, {select:selection}), true);
});

test('line breaks that trigger automatic semicolon insertion change the source contract', () => {
  assert.equal(matches(selection.replace('return token', 'return\ntoken'), {select:selection}), false);
  assert.equal(matches(selection.replace('return token', 'return /*\ncomment */ token'), {select:selection}), false);
});

test('named imports must bind the actual dependency from the expected module', () => {
  const match = source => contract.namedImportMatches?.(source, 'getSetting', './helpers.js') ?? false;
  assert.equal(match('import { getSetting, other } from "./helpers.js";'), true);
  for (const source of ['// import { getSetting } from "./helpers.js";', 'import { other as getSetting } from "./helpers.js";', 'import { getSetting } from "./foreign.js";']) assert.equal(match(source), false);
});
