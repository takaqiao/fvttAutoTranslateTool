import vm from 'node:vm';
import { installDsnQueueRecovery } from '../scripts/patches/dsn-queue.mjs';
export const settle = () => new Promise((resolve) => setImmediate(resolve));
export const observe = (promise) => {
  const state = { status: 'pending' };
  Promise.resolve(promise).then(
    (value) => Object.assign(state, { status: 'resolved', value }),
    (error) => Object.assign(state, { status: 'rejected', error })
  );
  return state;
};
export function setup(fixture,{ticker:nativeTicker}={}) {
  const errors = [],
    workers = [],
    ticks = [],
    landed = [];
  const listeners=[];
  const ticker=nativeTicker??{
    add(fn,box){ticks.push([fn,box]);listeners.push([fn,box]);return this;},
    remove(fn,box=null){for(let i=listeners.length-1;i>=0;i--)if(listeners[i][0]===fn&&listeners[i][1]===box)listeners.splice(i,1);return this;},
    has(fn){return listeners.some(([value])=>value===fn);},
    frame(){for(const[fn,box]of [...listeners])fn.call(box);},
    _cancelIfNeeded(){},
    get _head(){
      const head={next:null};let previous=head;
      for(const[fn,box]of listeners){
        const node={fn,context:box,next:null,destroy(){ticker.remove(fn,box);return this.next;}};
        previous.next=node;previous=node;
      }
      return head;
    }
  };
  let failure = null,
    release,workerRelease;
  const game = {
    settings: { get: (_module, key) => (key === 'maxDiceNumber' ? 20 : false) },
    dice3d: { pendingThrows: { noteBindsLanded: (binds) => landed.push(...binds) } }
  };
  const bo = {
    renderQueue: [],
    renderSFX() {},
    clearQueue(){},
    async playSFX() {
      if (failure === 'effects') {
        failure = null;
        throw Error('effects failed');
      }
      if (release) await release.promise;
    }
  };
  const context = vm.createContext({
    game,
    bo,
    r: { DICE_EVENT_TYPE: { RESULT: 1 } },
    setTimeout,
    clearTimeout,
    performance,
    DsnSettings: { isEnabled: () => true },
    DiceNotation: {
      mergeQueuedRollCommands: (items) =>
        items.flatMap((item) => item.params.throws.map((throwData) => [throwData]))
    },
    Utils: { removeTicker(fn){let node=ticker._head.next;while(node)node=node.fn===fn?node.destroy():node.next;} },
    canvas: {app:{ticker}}
  });
  const classes = vm.runInContext(
    '(()=>{' +
      fixture.accumulator +
      ';' +
      fixture.queue +
      ';' +
      fixture.boxClass +
      ';' +
      fixture.engineClass +
      ';return {AnimationQueue,DiceBox,ThrowEngine};})()',
    context
  );
  const worker = {
    async exec(name, args) {
      workers.push([name, args]);
      if(workerRelease&&((workerRelease.stage==='simulate'&&name==='simulateThrow')
        ||(workerRelease.stage==='collisions'&&name==='setCollisionResponse'&&args.enabled)
        ||(workerRelease.stage==='positions'&&name==='setBodyPositions')))await workerRelease.promise;
      if (
        (name === 'setCollisionResponse' && args.enabled && failure === 'collisions') ||
        (name === 'setBodyPositions' && failure === 'positions')
      ) {
        failure = null;
        throw Error(name + ' failed');
      }
      if (name === 'simulateThrow' && failure === 'simulate') {
        failure = null;
        throw Error('simulateThrow failed');
      }
      if (name === 'simulateThrow')
        return {
          ids: [],
          quaternionsBuffers: [],
          positionsBuffers: [],
          detectedCollides: [],
          deads: [],
          iterationsNeeded: 0,
          faceValues: {},
          finalQuaternions: {}
        };
      return true;
    }
  };
  const scene = { display: { innerWidth: 1000, innerHeight: 800 }, animatedDiceDetected: false },
    factory = { systems: new Map([['standard', { fire() {} }]]) };
  const engine = new classes.ThrowEngine(scene, worker, factory, {
    generateCollisionSounds: () => []
  });
  const box = Object.assign(Object.create(classes.DiceBox.prototype), {
    throwEngine: engine,
    physicsWorker: worker,
    diceScene: scene,
    dicefactory: factory,
    fadingDice: [],
    inputHandler: { clearPendingThrowDice() {}, updatePreRoll() {} },
    persistentDiceManager: {
      persistentDiceList: engine.persistentDiceList,
      updateRemoteAnimations() {}
    },
    renderScene() {}
  });
  const queue = new classes.AnimationQueue({
    canvasVisibility: { show() {}, hide() {} },
    pendingThrows: { noteBindsLanded: (binds) => landed.push(...binds) }
  });
  queue.attach(box);
  const install = () => installDsnQueueRecovery({ queue, ticker, game, recover: (error) => errors.push(error) });
  const enqueue = (count = 1) =>
    queue.enqueue(
      { throws: Array.from({ length: count }, () => ({ dice: [], dsnConfig: {} })) },
      {}
    );
  const finish = async () => {
    engine.iteration = 2;
    engine.minIterations = 0;
    engine.iterationsNeeded = 1;
    box.last_time = Date.now();
    box.isVisible = false;
    const context=ticks.at(-1)?.[1]??box;
    box.animateThrow.call(context);
    await settle();
  };
  const held = { id: 1, userData: { pendingBind: 'current' }, traverse() {} },
    other = {
      id: 2,
      userData: { constrained: true, pendingBind: 'other' },
      traverse() {},
      parent: { position: { x: 1, y: 2, z: 3 } }
    };
  return {
    context,
    classes,
    game,
    queue,
    engine,
    box,
    install,
    enqueue,
    finish,
    errors,
    workers,
    ticks,
    ticker,
    landed,
    held,
    other,
    fail: (value) => {
      failure = value;
    },
    holdWorker(stage){
      let resolve,reject;
      const promise=new Promise((r,j)=>{resolve=r;reject=j;});
      workerRelease={stage,promise};
      return Object.assign(resolve,{reject});
    },
    holdEffects() {
      let resolve, reject;
      const promise = new Promise((r, j) => {
        resolve = r;
        reject = j;
      });
      release = { promise };
      return Object.assign(resolve, { reject });
    }
  };
}
