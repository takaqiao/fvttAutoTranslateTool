// The source nameplates have different heights; retain each complete frame.
const nameplateRows = { cotct: [0, 269], sog: [269, 267], fotrp: [536, 267], av: [803, 266], bob: [1069, 333] };
// Painted rim centers in each 374 × 841 atlas column, excluding decorative tips.
const portraitCenters = { cotct: [200, 162], sog: [192, 165], fotrp: [188.5, 165.4], av: [182, 164], bob: [182, 170] };
const skin = (id, brass, ivory, talk, panel, hp, halo) => Object.freeze({
  id, frame: `frames/${id}-frame.png`, logo: `logos/${id}.webp`,
  title: id === 'cotct' ? 'logos/cotct-title.svg' : null,
  material: 'textures/focus-material-atlas.png',
  materialPosition: { cotct: '0%', sog: '25%', fotrp: '50%', av: '75%', bob: '100%' }[id],
  portraitCenter: Object.freeze({ x: `${portraitCenters[id][0] / 374 * 100}%`, y: `${portraitCenters[id][1] / 841 * 100}%` }),
  nameplate: 'textures/cast-nameplate-atlas.png',
  nameplateSize: `${1402 / nameplateRows[id][1] * 100}%`,
  nameplatePosition: `${nameplateRows[id][0] / (1402 - nameplateRows[id][1]) * 100}%`,
  palette: Object.freeze({ brass, ivory, talk, panel, hp, halo,
    'card-ink': { cotct: '#f1e4c8', sog: '#253e35', fotrp: '#482d1a', av: '#203d35', bob: '#303529' }[id],
    'card-muted': { cotct: '#cbb798', sog: '#3f5849', fotrp: '#69452b', av: '#345749', bob: '#555b49' }[id],
    'card-plaque': { cotct: '#673e3d', sog: '#d3c9a9', fotrp: '#e5c493', av: '#adc1ae', bob: '#c8c2a9' }[id],
  }),
});

export const SKINS = Object.freeze({
  cotct: skin('cotct', '#baa276', '#e7dfcb', '#f1d9a7', '#29242ded', '#c18787', '#e8c28266'),
  sog: skin('sog', '#b7a876', '#e3e4cf', '#b8e2ca', '#22352fed', '#9eb6a0', '#acd4bb66'),
  fotrp: skin('fotrp', '#d0a66c', '#efe0c7', '#ffdda0', '#3d2c24ed', '#cc9273', '#f3b66966'),
  av: skin('av', '#8ea58f', '#e0e5d7', '#a6e4c6', '#22382fed', '#87b5a1', '#8cccad66'),
  bob: skin('bob', '#b9a272', '#e8e2cd', '#e5d59c', '#2d3b30ed', '#a1b596', '#c0c99166'),
});
export const WORLD_SKINS = Object.freeze({ cotct: 'cotct', sog: 'sog', pnvfcgjbf2cjp7gz: 'fotrp', '-': 'av', ujx5r8oipw7ercdr: 'bob' });

/** Only the approved five treatments can supply image paths or CSS colors. */
export function resolveSkin(id) {
  if (typeof id !== 'string') return null;
  if (Object.hasOwn(SKINS, id)) return SKINS[id];
  return Object.hasOwn(WORLD_SKINS, id) ? SKINS[WORLD_SKINS[id]] : null;
}
