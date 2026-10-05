const paths = Object.freeze({
  shield: 'M12 2 21 6v6c0 5-6 9-9 10-3-1-9-5-9-10V6Z',
  fortitude: 'M5 20v-8l3-2V5h3v5l2-1V3h3v8l3 2v7Z',
  reflex: 'M14 2 5 13h6l-1 9 9-13h-6Z',
  will: 'M12 2 22 12 12 22 2 12Zm0 5v10m-5-5h10',
  eye: 'M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Zm10-3a3 3 0 1 0 0 6 3 3 0 0 0 0-6',
  boot: 'M6 3h7v9l6 3v5H3v-5l3-3Z',
  wing: 'M3 19C3 8 10 3 22 3l-5 6h-4l-3 4h5l-6 6Z',
  hero: 'm12 2 3 7 7 1-5 5 1 7-6-4-6 4 1-7-5-5 7-1Z',
  focus: 'M12 2v4m0 12v4M2 12h4m12 0h4M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10',
  dc: 'M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm0 5a5 5 0 1 0 0 10 5 5 0 0 0 0-10m0 3v4m-2-2h4',
  temporary: 'M12 2 21 6v6c0 5-6 9-9 10-3-1-9-5-9-10V6Zm0 6v8m-4-4h8',
  pool: 'M4 6h16v14H4Zm4-4v8m8-8v8M8 14h8',
  hardness: 'M5 3h14l3 7-10 12L2 10Zm-3 7h20M5 3l7 19 7-19',
  raised: 'M12 2 21 6v6c0 5-6 9-9 10-3-1-9-5-9-10V6Zm0 15V7m-4 4 4-4 4 4',
  broken: 'M12 2 21 6v6c0 5-6 9-9 10-3-1-9-5-9-10V6Zm1 2-4 6 5 3-3 7',
});

// PF2e HUD's statistic mappings use Free glyphs already loaded by Foundry core.
const nativeClasses = Object.freeze({
  shield: 'fa-shield', fortitude: 'fa-chess-rook', reflex: 'fa-person-running',
  will: 'fa-brain', eye: 'fa-eye', boot: 'fa-shoe-prints', wing: 'fa-feather',
});

export function hudIcon(document, name, label) {
  const icon = document.createElement('span');
  icon.className = 'hud-icon';
  icon.setAttribute('role', 'img');
  icon.setAttribute('aria-label', label);
  if (Object.hasOwn(nativeClasses, name)) {
    icon.classList.add('has-native-icon');
    const native = document.createElement('i');
    native.className = `hud-icon-native fa-solid ${nativeClasses[name]}`;
    native.setAttribute('aria-hidden', 'true');
    icon.append(native);
  }
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'hud-icon-fallback');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('aria-hidden', 'true');
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', Object.hasOwn(paths, name) ? paths[name] : paths.pool);
  svg.append(path);
  icon.append(svg);
  return icon;
}
