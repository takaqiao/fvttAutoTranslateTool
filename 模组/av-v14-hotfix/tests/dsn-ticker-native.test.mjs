import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {setup,observe,settle} from './dsn-queue-harness.mjs';

const nativeApp=process.env.FVTT_NATIVE_APP??'C:/Program Files/Foundry Virtual Tabletop/resources/app';
const file=path.join(nativeApp,'node_modules/pixi.js/dist/pixi.mjs');
const fixture=JSON.parse(fs.readFileSync(new URL('./fixtures/dsn-queue-6.4.3-native.json',import.meta.url)));
test('actual Foundry Pixi ticker preserves owned add/remove/context/restore identities',
  {skip:!fs.existsSync(file)&&'Set FVTT_NATIVE_APP to the installed Foundry resources/app'},async()=>{
    const {Ticker}=await import(pathToFileURL(file));
    const ticker=new Ticker(),f=setup(fixture,{ticker}),native=f.box.animateThrow;
    ticker.add(native,f.box);
    const patch=f.install(),wrapped=f.box.animateThrow;
    assert.notEqual(wrapped,native);assert.equal(ticker.count,1);
    assert.equal(ticker._head.next.fn,wrapped);assert.equal(ticker._head.next.context,f.box);
    const state=observe(f.enqueue());await settle();assert.equal(ticker.count,1);
    f.engine.persistentDiceList.push(f.other);f.other.userData.persistentId='older';
    f.box._startMeshFade=()=>{};f.box.persistentDiceManager.removePersistentDie=async()=>true;
    await f.box.fadeOutPersistentDie('older',1000);
    assert.equal(ticker.count,1);assert.equal(ticker._head.next.fn,wrapped);
    patch.restore();assert.equal(ticker.count,1);assert.equal(ticker._head.next.fn,native);
    f.engine.persistentDiceList.length=0;
    await f.finish();assert.equal(state.value,true);await f.queue.idle();assert.equal(ticker.count,0);
    ticker.destroy();
  });
