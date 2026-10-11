import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createGamePause} from '../../scripts/exploration/game-pause.mjs';

function fixture({calendar=false,paused=false}={}){
 const gm={id:'G',isGM:true,active:true},listeners=new Set(),calls=[];
 const game={user:gm,users:{activeGM:gm},paused,modules:new Map(calendar?[['calendaria',{active:true}]]:[]),time:{worldTime:-100}};
 const clock={running:calendar&&!paused,predictedWorldTime:-100,ADVANCE_INTERVAL_MS:60000};
 const Hooks={on:(event,fn)=>{assert.equal(event,'updateWorldTime');listeners.add(fn);return fn},off:(event,fn)=>listeners.delete(fn)};
 game.togglePause=(value,options)=>{calls.push([value,options]);game.paused=value;clock.running=false;return value};
 const pause=createGamePause({game,Hooks,getClock:()=>clock,timeoutMs:40});
 return {game,clock,Hooks,calls,listeners,pause,emit:(dt=9,userId='G')=>{for(const fn of listeners)fn(game.time.worldTime,dt,{},userId)}};
}

test('automatic pause broadcasts the GM pause and leaves world time unchanged',async()=>{
 const f=fixture();await f.pause();assert.equal(f.game.paused,true);assert.equal(f.game.time.worldTime,-100);assert.deepEqual(f.calls,[[true,{broadcast:true}]]);assert.equal(f.listeners.size,0);
});
test('an already paused game does not broadcast another pause',async()=>{
 const f=fixture({paused:true});await f.pause();assert.deepEqual(f.calls,[]);assert.equal(f.game.paused,true);
});
test('pause waits for Calendaria accumulated time and checks after its update hook',async()=>{
 const f=fixture({calendar:true}),toggle=f.game.togglePause;let finished=false;
 f.game.togglePause=(...args)=>{const result=toggle(...args);f.clock.predictedWorldTime=-91;return result};
 const pending=f.pause().then(()=>{finished=true});await Promise.resolve();assert.equal(finished,false);
 f.game.time.worldTime=-91;f.clock.predictedWorldTime=-82;f.emit();f.clock.predictedWorldTime=-91;
 await pending;assert.equal(finished,true);assert.equal(f.listeners.size,0);assert.equal(f.game.paused,true);
});
test('an already paused calendar still waits for its in-flight time commit',async()=>{
 const f=fixture({calendar:true,paused:true});f.clock.predictedWorldTime=-91;
 const pending=f.pause();pending.catch(()=>{});await Promise.resolve();assert.equal(f.listeners.size,1);
 f.game.time.worldTime=-91;f.emit();await pending;assert.deepEqual(f.calls,[]);assert.equal(f.listeners.size,0);
});
test('unconfirmed calendar time times out and removes the waiting hook',async()=>{
 const f=fixture({calendar:true});f.clock.predictedWorldTime=-91;
 await assert.rejects(f.pause(),/exploration-clock-pause-unconfirmed/);assert.equal(f.listeners.size,0);assert.equal(f.game.paused,true);
});
test('a calendar that keeps running cannot start automatic recovery',async()=>{
 const f=fixture({calendar:true});f.game.togglePause=value=>{f.game.paused=value;return value};
 await assert.rejects(f.pause(),/exploration-clock-still-running/);assert.equal(f.listeners.size,0);
});
test('losing active GM while waiting cannot finish the pause boundary',async()=>{
 const f=fixture({calendar:true});f.clock.predictedWorldTime=-91;const pending=f.pause();
 f.game.users.activeGM={id:'other'};f.game.time.worldTime=-91;f.emit();
 await assert.rejects(pending,/active-gm-required/);assert.equal(f.listeners.size,0);
});
test('a missing calendar clock and a failed game pause reject without native work',async()=>{
 const f=fixture({calendar:true});const pause=createGamePause({game:f.game,Hooks:f.Hooks,getClock:()=>undefined});
 await assert.rejects(pause(),/exploration-clock-pause-unavailable/);assert.equal(f.calls.length,0);
 const plain=fixture();plain.game.togglePause=()=>false;await assert.rejects(plain.pause(),/game-pause-failed/);
});
test('nonactive GM cannot pause even a currently running game',async()=>{
 const f=fixture();f.game.users.activeGM={id:'other'};
 await assert.rejects(f.pause(),/active-gm-required/);assert.equal(f.calls.length,0);
});
test('an unrelated clock update cannot impersonate a pending calendar flush',async()=>{
 const f=fixture({calendar:true});f.clock.predictedWorldTime=-91;const pending=f.pause();
 f.game.time.worldTime=500;f.clock.predictedWorldTime=500;f.emit(600);
 await assert.rejects(pending,/external-world-time-change/);assert.equal(f.listeners.size,0);
});
test('the calendar direct mode cannot claim that its invisible pending ticks settled',async()=>{
 const f=fixture({calendar:true});f.clock.ADVANCE_INTERVAL_MS=1000;
 await assert.rejects(f.pause(),/exploration-clock-pause-unavailable/);assert.equal(f.game.paused,true);assert.equal(f.listeners.size,0);
});
