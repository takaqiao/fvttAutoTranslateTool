import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';

const sha256=bytes=>createHash('sha256').update(bytes).digest('hex');
function replaceOnce(source,before,after) {
  if(source.split(before).length!==2)throw new Error(`Expected one source anchor: ${before.slice(0,90)}`);
  return source.replace(before,after);
}

export async function buildCandidate(dataRoot,outputRoot) {
  await fs.mkdir(outputRoot,{recursive:false});
  const files=[];
  async function write(relative,transform) {
    const before=await fs.readFile(path.join(dataRoot,relative));
    const source=before.toString('utf8'),newline=source.includes('\r\n')?'\r\n':'\n';
    const result=transform(source.replaceAll('\r\n','\n')).replaceAll('\n',newline);
    const after=Buffer.from(result,'utf8'),target=path.join(outputRoot,'Data',relative);
    await fs.mkdir(path.dirname(target),{recursive:true});await fs.writeFile(target,after);
    files.push({path:relative,before:sha256(before),after:sha256(after)});
  }
  await write('modules/cotct-music-guard/bootstrap.mjs',source=>{
    source=replaceOnce(source,"import {catalog} from './managed-catalog.mjs';", "import {catalog} from './managed-catalog.mjs';\nimport {createTargetedAudio} from './targeted-audio.mjs';");
    return replaceOnce(source,'  const component=createGuard(',`  Hooks.once('setup',()=>{
    try{
      currentGame.cotctTargetedAudio=createTargetedAudio({game:currentGame,
        paths:[...catalog.regular.flatMap(playlist=>playlist.sounds.map(sound=>sound.path)),
          ...catalog.soundscape.sounds.map(sound=>sound.path)],onError:report});
    }catch(error){report(error);}
  });
  const component=createGuard(`);
  });
  await write('assets/cotct-music/runtime/player-api.mjs',source=>replaceOnce(source,`    async stop() {
      requireStop();
      cancelPending();
      return controller.stop();
    }`, `    async stop({ targeted = true } = {}) {
      requireStop();
      cancelPending();
      const stops = [controller.stop()];
      if (targeted && boundGame.cotctTargetedAudio) {
        stops.push(Promise.resolve().then(() => boundGame.cotctTargetedAudio.stop()));
      }
      const results = await Promise.allSettled(stops);
      const errors = results.filter(result => result.status === 'rejected').map(result => result.reason);
      if (errors.length) throw new AggregateError(errors, 'Could not stop every managed audio source');
      return results[0].value;
    }`));
  await write('modules/cotct-full-moods/src/main.mjs',source=>{
    source=replaceOnce(source,"button.disabled=!control||(!status.run&&!pendingPlay);", "button.disabled=!control||(!status.run&&!pendingPlay&&!game.cotctTargetedAudio?.hasActive());");
    source=replaceOnce(source,'async function stopLegacy(canContinue){','async function stopLegacy(canContinue,{targeted=true}={}){');
    source=replaceOnce(source,'await player.stop();return canContinue();','await player.stop({targeted});return canContinue();');
    source=replaceOnce(source,'if(!await stopLegacy(()=>current(op))||!current(op))', 'if(!await stopLegacy(()=>current(op),{targeted:false})||!current(op))');
    return replaceOnce(source,`    if(clock.ready)controller.stop();else controller.disconnect();
    updateMixControls();await stopLegacy(()=>current(op));`, `    const errors=[];
    try{if(clock.ready)controller.stop();else controller.disconnect();}catch(error){errors.push(error);}
    updateMixControls();
    try{await stopLegacy(()=>current(op));}catch(error){errors.push(error);}
    if(errors.length)throw new AggregateError(errors,'Could not stop every managed audio source');`);
  });
  for(const [module,from,to] of [['cotct-music-guard','0.1.1','0.1.2'],['cotct-full-moods','0.3.1','0.3.2']]) {
    await write(`modules/${module}/module.json`,source=>{
      if(JSON.parse(source).version!==from)throw new Error(`Unexpected ${module} base version`);
      return replaceOnce(source,`"version": "${from}"`,`"version": "${to}"`);
    });
  }
  const relative='modules/cotct-music-guard/targeted-audio.mjs';
  try{await fs.access(path.join(dataRoot,relative));throw new Error('Targeted observer already exists');}
  catch(error){if(error.code!=='ENOENT')throw error;}
  const helper=await fs.readFile(new URL('../src/targeted-audio.mjs',import.meta.url));
  await fs.writeFile(path.join(outputRoot,'Data',relative),helper);
  files.push({path:relative,before:null,after:sha256(helper)});
  const manifest={schema:1,release:'cotct-audio-compat-v0.1.0',kind:'file-delta',files};
  await fs.writeFile(path.join(outputRoot,'manifest.json'),JSON.stringify(manifest,null,2)+'\n','utf8');
  return manifest;
}

if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  const [dataRoot,outputRoot]=process.argv.slice(2);
  if(!dataRoot||!outputRoot)throw new Error('Usage: node tools/build-candidate.mjs EXISTING_DATA NEW_OUTPUT');
  console.log(JSON.stringify(await buildCandidate(path.resolve(dataRoot),path.resolve(outputRoot)),null,2));
}
