import {isActiveGM} from './document-store.mjs';

export function createGamePause({game,Hooks,getClock=()=>globalThis.CALENDARIA?.managers?.TimeClock,timeoutMs=5000}){
 return async()=>{
  const user=game.user;
  const authority=()=>{if(game.user!==user||!isActiveGM(game)||user.active===false)throw Error('active-gm-required')};
  authority();
  const calendar=game.modules.get('calendaria')?.active===true,clock=calendar?getClock():null;
  if(calendar&&!clock)throw Error('exploration-clock-pause-unavailable');
  const pause=()=>{if(game.paused!==true)try{game.togglePause(true,{broadcast:true})}catch(cause){throw Error('game-pause-failed',{cause})}authority();if(game.paused!==true)throw Error('game-pause-failed')};
  if(!clock){pause();return}
  await new Promise((resolve,reject)=>{
   let finished=false,timer,hook,from,expected;
   const finish=error=>{if(finished)return;finished=true;clearTimeout(timer);if(hook!==undefined)Hooks.off('updateWorldTime',hook);if(error)reject(error);else resolve()};
   const check=()=>{
    authority();if(game.paused!==true)throw Error('game-pause-failed');
    if(getClock()!==clock||game.modules.get('calendaria')?.active!==true)throw Error('exploration-clock-pause-unavailable');
    if(clock.running!==false)throw Error('exploration-clock-still-running');
    if(!Number.isFinite(game.time.worldTime)||!Number.isFinite(clock.predictedWorldTime))throw Error('exploration-clock-pause-unavailable');
   };
   try{
    hook=Hooks.on('updateWorldTime',(worldTime,dt,options,userId)=>{
     // Calendaria retires pending seconds in its own updateWorldTime listener.
     queueMicrotask(()=>{if(finished)return;try{
      check();if(dt===0)return;
      if(worldTime!==expected||dt!==expected-from||userId!==user.id||game.time.worldTime!==expected)throw Error('external-world-time-change');
      if(clock.predictedWorldTime===expected)finish();
     }catch(error){finish(error)}});
    });
    pause();check();
    // The one-second mode does not expose pending commits through this getter.
    if(!Number.isFinite(clock.ADVANCE_INTERVAL_MS)||clock.ADVANCE_INTERVAL_MS<=1000)throw Error('exploration-clock-pause-unavailable');
    from=game.time.worldTime;expected=clock.predictedWorldTime;
    if(expected===from){finish();return}
    if(expected<from)throw Error('exploration-clock-pause-unavailable');
    timer=setTimeout(()=>finish(Error('exploration-clock-pause-unconfirmed')),timeoutMs);
   }catch(error){finish(error)}
  });
 };
}
