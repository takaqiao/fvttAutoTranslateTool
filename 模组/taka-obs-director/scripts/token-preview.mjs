const primitive = value => typeof value === 'string' || typeof value === 'boolean'
  || (typeof value === 'number' && Number.isFinite(value)) ? value : null;

function appearance(token) {
  const texture = token.texture, ring = token.ring;
  return JSON.stringify([
    token.width, token.height, token.rotation, token.lockRotation, token.alpha,
    texture?.src, texture?.fit, texture?.scaleX, texture?.scaleY, texture?.tint, texture?.alphaThreshold,
    texture?.anchorX, texture?.anchorY, texture?.offsetX, texture?.offsetY,
    ring?.enabled, ring?.subject?.texture, ring?.subject?.scale,
    ring?.colors?.ring, ring?.colors?.background, ring?.effects,
  ].map(primitive));
}

/** Snapshot the scene mesh before handing a detached sprite to ImageHelper. */
export function createNativeTokenThumbnail({ renderer, imageHelper, Sprite } = {}) {
  if (typeof renderer?.generateTexture !== 'function' || typeof imageHelper?.createThumbnail !== 'function'
    || typeof Sprite !== 'function') return undefined;
  return async (mesh, options) => {
    if (typeof mesh?.getLocalBounds !== 'function') return null;
    const bounds = mesh.getLocalBounds();
    if (!Number.isFinite(bounds?.width) || bounds.width <= 0
      || !Number.isFinite(bounds?.height) || bounds.height <= 0) return null;
    const resolution = Math.min(1, 512 / Math.max(bounds.width, bounds.height));
    const texture = renderer.generateTexture(mesh, { resolution });
    let sprite;
    try {
      sprite = new Sprite(texture);
      return await imageHelper.createThumbnail(sprite, options);
    } finally {
      try { sprite?.destroy({ texture: false, baseTexture: false }); }
      finally { texture.destroy(true); }
    }
  };
}

/** Keep one public token's native mesh thumbnail, without reading its Actor. */
export function createTokenPreview({ createThumbnail, canDisplay, onChange } = {}) {
  let current = null, disposed = false, warned = false;
  function allowed(token) {
    if (disposed || !token) return false;
    try { return canDisplay?.(token) === true; } catch { return false; }
  }
  function warn() {
    if (warned) return;
    warned = true;
    console.warn('Taka OBS Director could not create the token preview.');
  }
  function clear() { current = null; }
  function request(token) {
    if (!allowed(token)) { clear(); return null; }
    if (current?.token !== token) clear();
    if (typeof createThumbnail !== 'function') return null;
    try {
      const mesh = token.object?.mesh;
      if (!mesh?.getBounds) { clear(); return null; }
      const key = appearance(token);
      if (current?.token === token && current.mesh === mesh && current.key === key) return current.image;
      const bounds = mesh.getBounds();
      if (!Number.isFinite(bounds?.width) || bounds.width <= 0
        || !Number.isFinite(bounds?.height) || bounds.height <= 0) { clear(); return null; }
      const scale = 512 / Math.max(bounds.width, bounds.height);
      const options = {
        format: 'image/png',
        center: true,
        width: Math.max(1, Math.round(bounds.width * scale)),
        height: Math.max(1, Math.round(bounds.height * scale)),
      };
      const entry = { token, mesh, key, image: null };
      current = entry;
      let exported;
      try { exported = createThumbnail(mesh, options); } catch { warn(); return null; }
      Promise.resolve(exported).then(result => {
        if (current !== entry || disposed) return;
        if (!allowed(token)) { clear(); return; }
        if (token.object?.mesh !== mesh || appearance(token) !== key) { clear(); return; }
        const image = result?.thumb;
        if (typeof image !== 'string' || !/^data:image\/png;base64,[A-Za-z0-9+/]+={0,2}$/.test(image)) return;
        entry.image = image;
        onChange?.(image);
      }).catch(() => { if (current === entry && !disposed) warn(); });
      return null;
    } catch {
      clear();
      warn();
      return null;
    }
  }
  function invalidate(token) {
    if (current?.token === token) clear();
  }
  function peek(token) {
    if (!allowed(token) || current?.token !== token) { clear(); return null; }
    return current.image;
  }
  function dispose() { disposed = true; clear(); }
  return { request, peek, invalidate, clear, dispose };
}
