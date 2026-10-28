const adapted = new WeakSet();
const nestedOverrides = new WeakSet();
const SAVE_METHODS = ['saveToken', 'saveAvatar', 'saveBoth', 'saveHeadless'];

function namedOverrides(editor, overrides, mode, version) {
  if (mode === 'token' && editor.isWildcard && !overrides?.filename) {
    const pattern = editor.actor?.prototypeToken?.texture?.src?.split('?')[0].split('/').at(-1);
    if (pattern?.includes('*')) return {...overrides, filename: pattern.replaceAll('*', `token-${version}`)};
  }
  const original = overrides?.filename || editor.actor?.id || 'tokenizer';
  const match = original.match(/^(.*?)(\.[^./\\]+)?$/);
  return {...overrides, filename: `${match[1]}-${mode}-${version}${match[2] ?? ''}`};
}

/** Tokenizer 2.1.0 editorOpen 提供的实例接缝；未发该 hook 的批量/API 保存不经过这里。 */
export function adaptTokenizerEditor(editor) {
  const save = editor?.save;
  if (!save || SAVE_METHODS.some(method => typeof save[method] !== 'function')) return false;
  if (adapted.has(save)) return true;
  for (const [method, mode] of [['saveToken', 'token'], ['saveAvatar', 'avatar']]) {
    const original = save[method];
    save[method] = function (options = {}, ...rest) {
      return original.call(this, {...options,
        overrides: namedOverrides(editor, options.overrides, mode, crypto.randomUUID()),
      }, ...rest);
    };
  }
  const both = save.saveBoth;
  save.saveBoth = function (options = {}, ...rest) {
    const version = crypto.randomUUID();
    const overrides = {...options.overrides};
    for (const mode of ['token', 'avatar']) {
      overrides[mode] = namedOverrides(editor, overrides[mode], mode, version);
      nestedOverrides.add(overrides[mode]);
    }
    return both.call(this, {...options, overrides}, ...rest);
  };
  const headless = save.saveHeadless;
  save.saveHeadless = function (mode, layers, options = {}, ...rest) {
    if (!['token', 'avatar'].includes(mode)) throw new Error(`不支持的 Tokenizer 保存模式：${mode}`);
    // saveBoth 传入已分配的对象；无需共享异步状态，也不会给嵌套调用再加一次后缀。
    const overrides = nestedOverrides.has(options.overrides) ? options.overrides
      : namedOverrides(editor, options.overrides, mode, crypto.randomUUID());
    return headless.call(this, mode, layers, {...options, overrides}, ...rest);
  };
  adapted.add(save);
  return true;
}

export function installTokenizerSaveAdapter() {
  Hooks.on('tokenizer-2.editorOpen', ({editor}) => {
    if (adaptTokenizerEditor(editor)) return;
    console.warn('taka-charsync | Tokenizer 保存接口不兼容，未安装独立文件名适配');
    ui.notifications.warn('当前 Tokenizer 保存接口不兼容；再次保存可能覆盖同名原图。提交时仍会校验并冻结本地媒体。');
  });
}
