// Read source without evaluating it. Literals stay opaque, so quoted examples
// and comments cannot stand in for the declarations a patch depends on.
const REGEXP_PREFIXES = new Set([
  '(', '[', '{', ',', ';', ':', '?', '=>', '...',
  '=', '+', '-', '*', '/', '%', '**', '&', '|', '^', '~', '!', '&&', '||', '??',
  '<', '>', '<=', '>=', '==', '!=', '===', '!==', '<<', '>>', '>>>',
  '+=', '-=', '*=', '/=', '%=', '**=', '&=', '|=', '^=', '&&=', '||=', '??=',
  'return', 'throw', 'case', 'yield', 'await', 'typeof', 'void', 'delete',
  'in', 'instanceof', 'of', 'else', 'do', 'new', 'extends'
]);
const NON_EXPRESSION_WORDS = new Set([
  'await', 'break', 'case', 'catch', 'class', 'const', 'continue', 'debugger',
  'default', 'delete', 'do', 'else', 'enum', 'export', 'extends', 'finally',
  'for', 'function', 'if', 'import', 'in', 'instanceof', 'let', 'new', 'return',
  'super', 'switch', 'throw', 'try', 'typeof', 'var', 'void', 'while', 'with',
  'yield', 'implements', 'interface', 'package', 'private', 'protected', 'public',
  'static', 'as', 'async', 'from', 'get', 'of', 'set', 'using'
]);

export function sourceTokens(source) {
  if (typeof source !== 'string') throw new TypeError('Expected JavaScript source');
  let position = 0;
  const quoted = quote => {
    position++;
    while (position < source.length) {
      const character = source[position++];
      if (character === '\\') {position++; continue;}
      if (character === quote) return;
      if (character === '\n' || character === '\r') break;
    }
    throw new SyntaxError('Unterminated string');
  };
  const template = () => {
    position++;
    while (position < source.length) {
      const character = source[position++];
      if (character === '\\') {position++; continue;}
      if (character === '`') return;
      if (character !== '$' || source[position] !== '{') continue;
      position++;
      let level = 1, previous;
      while (level) {
        const token = next(previous);
        if (!token) throw new SyntaxError('Unterminated template expression');
        if (token.value === '{') level++;
        if (token.value === '}') level--;
        previous = token;
      }
    }
    throw new SyntaxError('Unterminated template');
  };
  const next = previous => {
    while (position < source.length) {
      if (/\s/.test(source[position])) {position++; continue;}
      if (source.startsWith('//', position)) {
        const end = source.slice(position + 2).search(/[\r\n\u2028\u2029]/);
        position = end < 0 ? source.length : position + 2 + end;
        continue;
      }
      if (source.startsWith('/*', position)) {
        const end = source.indexOf('*/', position + 2);
        if (end < 0) throw new SyntaxError('Unterminated comment');
        position = end + 2;
        continue;
      }
      break;
    }
    if (position === source.length) return null;
    if (source.startsWith('<!--', position) || source.startsWith('-->', position)) throw new SyntaxError('Unsupported legacy comment');
    const start = position, character = source[position];
    let kind = 'punctuator';
    if (character === '"' || character === "'") {quoted(character); kind = 'literal';}
    else if (character === '`') {template(); kind = 'literal';}
    else if (character === '/') {
      const expressionStart = !previous || REGEXP_PREFIXES.has(previous.value);
      const expressionEnd = previous && !/[\r\n\u2028\u2029]/.test(source.slice(previous.end, start))
        && (previous.kind === 'literal' || previous.value === ']'
          || (previous.kind === 'identifier' && !NON_EXPRESSION_WORDS.has(previous.value)));
      // Only a proven expression end permits division. Unknown statement and
      // keyword contexts, including an ASI boundary, must not expose regexp contents.
      if (!expressionStart && !expressionEnd) throw new SyntaxError('Ambiguous slash');
      if (expressionStart) {
        kind = 'literal';
        position++;
        let inClass = false, closed = false;
        while (position < source.length) {
          const value = source[position++];
          if (value === '\\') {position++; continue;}
          if (/[\n\r\u2028\u2029]/.test(value)) break;
          if (value === '[') inClass = true;
          if (value === ']') inClass = false;
          if (value === '/' && !inClass) {closed = true; break;}
        }
        if (!closed) throw new SyntaxError('Unterminated regexp');
        while (/[a-z]/i.test(source[position] ?? '') && position < source.length) position++;
      } else position += source[position + 1] === '=' ? 2 : 1;
    } else {
      const value = source.slice(position).match(/^(?:[A-Za-z_$][\w$]*|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|===|!==|>>>|\*\*=|&&=|\|\|=|\?\?=|=>|==|!=|<=|>=|\+\+|--|&&|\|\||\?\?|\?\.|\*\*|<<|>>|[+*%&|^!?=<>-]=|\.\.\.|[{}()[\];,.?:~+*%&|^!<>=-])/);
      if (!value) throw new SyntaxError('Unsupported source token');
      position += value[0].length;
      if (/^[A-Za-z_$]/.test(value[0])) kind = 'identifier';
      else if (/^(?:\d|\.\d)/.test(value[0])) kind = 'literal';
    }
    return {value:source.slice(start, position), start, end:position, kind};
  };
  const tokens = [], stack = [];
  let token;
  while ((token = next(tokens.at(-1)))) {
    const previous = tokens.at(-1);
    if (previous && /[\r\n\u2028\u2029]/.test(source.slice(previous.end, token.start))
      && (['return', 'throw', 'break', 'continue', 'yield', 'async'].includes(previous.value)
        || ['++', '--'].includes(token.value))) {
      tokens.push({value:'\n', start:previous.end, end:token.start, depth:stack.length});
    }
    const close = token.value === '}' ? '{' : token.value === ')' ? '(' : token.value === ']' ? '[' : null;
    if (close) {
      const opening = stack.pop();
      if (opening === undefined || tokens[opening].value !== close) throw new SyntaxError(`Unbalanced source at ${token.start}`);
      token.match = opening;
      tokens[opening].match = tokens.length;
    }
    token.depth = stack.length;
    tokens.push(token);
    if (['{', '(', '['].includes(token.value)) stack.push(tokens.length - 1);
  }
  if (stack.length) throw new SyntaxError('Unbalanced source');
  return tokens;
}

function declaration(tokens, name) {
  const found = [];
  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index];
    if (token.depth !== 0 || tokens[index + 1]?.value !== name) continue;
    const start = token.value === 'function' && tokens[index - 1]?.value === 'async' ? index - 1 : index;
    if (start && ![';', '}', 'export'].includes(tokens[start - 1].value)) continue;
    let end;
    if (token.value === 'function' && tokens[index + 2]?.value === '(') {
      const body = tokens[index + 2].match + 1;
      if (tokens[body]?.value === '{') end = tokens[body].match;
    } else if (token.value === 'const' && tokens[index + 2]?.value === '=') {
      end = tokens.findIndex((entry, offset) => offset > index && entry.depth === 0 && entry.value === ';');
      if (end < 0) end = undefined;
    }
    if (end !== undefined) found.push({start, end, nameIndex:index + 1});
  }
  if (found.length !== 1) return null;
  const range = found[0];
  for (let index = 0; index < tokens.length; index++) {
    if (tokens[index].value !== name || index === range.nameIndex || ['.', '?.'].includes(tokens[index - 1]?.value)) continue;
    if (/^(?:=|\+=|-=|\*=|\/=|%=|&&=|\|\|=|\?\?=|\+\+|--)$/.test(tokens[index + 1]?.value)
      || ['++', '--'].includes(tokens[index - 1]?.value)) return null;
  }
  return range;
}

export function extractDeclaration(source, name) {
  try {
    const tokens = sourceTokens(source), range = declaration(tokens, name);
    return range ? source.slice(tokens[range.start].start, tokens[range.end].end) : null;
  } catch { return null; }
}

export function matchesDeclarations(source, expected) {
  try {
    const tokens = sourceTokens(source);
    return Object.entries(expected).every(([name, alternatives]) => {
      const range = declaration(tokens, name);
      if (!range) return false;
      const actual = JSON.stringify(tokens.slice(range.start, range.end + 1).map(token => token.value));
      return [alternatives].flat().some(value => {
        const normalized = extractDeclaration(value, name);
        return normalized && JSON.stringify(sourceTokens(normalized).map(token => token.value)) === actual;
      });
    });
  } catch { return false; }
}

export function namedImportMatches(source, name, path) {
  try {
    const tokens = sourceTokens(source);
    let matches = 0;
    for (let index = 0; index < tokens.length; index++) {
      if (tokens[index].depth !== 0 || tokens[index].value !== 'import' || tokens[index + 1]?.value !== '{') continue;
      const end = tokens[index + 1].match;
      if (tokens[end + 1]?.value !== 'from' || ![JSON.stringify(path), `'${path}'`].includes(tokens[end + 2]?.value)) continue;
      for (let offset = index + 2; offset < end; offset++) {
        if (tokens[offset].value === name && [',', '{'].includes(tokens[offset - 1]?.value)
          && [',', '}'].includes(tokens[offset + 1]?.value)) matches++;
      }
    }
    return matches === 1;
  } catch { return false; }
}
