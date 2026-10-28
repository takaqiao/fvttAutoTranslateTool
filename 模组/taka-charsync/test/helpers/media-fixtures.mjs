import {snapshotActorMedia} from '../../scripts/foundry/media.mjs';
import {collectMediaReferences} from '../../scripts/lib/media.mjs';

/** 同 URL 的各侧 fixture 提供相同字节；分叉字节由 live-media 回归单独设置。 */
export async function freezeMediaFixtures(docs) {
  const files = new Map(docs.flatMap(collectMediaReferences).map(({src}) =>
    [src.replace(/[?#].*$/s, ''), new TextEncoder().encode(src.replace(/[?#].*$/s, ''))]));
  const io = {read: async path => files.get(path), store: async (path, data) => files.set(path, data)};
  const frozen = [];
  for (const doc of docs) frozen.push((await snapshotActorMedia([doc], io)).docs[0]);
  const fetchPublished = globalThis.fetch;
  globalThis.fetch = async url => {
    if (url.startsWith('published/')) return fetchPublished(url);
    const data = files.get(decodeURIComponent(url));
    return {ok: Boolean(data), status: data ? 200 : 404, arrayBuffer: async () => data};
  };
  return frozen;
}
