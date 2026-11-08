// pf2e-hud's floating tracker renders in full whenever the core combat tracker
// does, which core asks for on every actor update of a combatant. Each pass
// replaced the whole tracker and then read layout to scroll to the current turn.
const MODULE='pf2e-hud';
const ID='pf2e-hud-tracker';
const adopted=new WeakSet();

// Only argument shapes that mean "refresh with current state" are merged.
function plain(options){
  if(options===undefined||typeof options==='boolean')return true;
  return !!options&&Object.getPrototypeOf(options)===Object.prototype&&Object.keys(options).every(key=>key==='force');
}

// The identical-content skip relies on the tracker replacing all of its content
// from the rendered string.
function replacesContent(fn){
  const source=Reflect.apply(Function.prototype.toString,fn,[]);
  const [,result,content]=source.match(/^[^(]*\(\s*([\w$]+)\s*,\s*([\w$]+)/)??[];
  if(!result)return false;
  const name=value=>value.replaceAll('$','\\$');
  return new RegExp(`(^|[^\\w$.])${name(content)}\\.innerHTML\\s*=\\s*${name(result)}(?![\\w$])`).test(source);
}

/** Merge bursts of full tracker refreshes and keep unchanged tracker DOM. */
export function installHudTrackerPatch({runtime=globalThis,report=()=>{},windowMs=100}={}){
  const stats={rendered:0,throttled:0,dropped:0,identical:0};
  const finish=(status,extra={})=>{const result={feature:'hudTracker',status,stats,...extra};report(result);return result;};
  if(!runtime.game?.modules?.get(MODULE)?.active)return finish('inactive',{reason:'module-inactive'});
  const {Hooks}=runtime;
  if(typeof Hooks?.on!=='function'||typeof Hooks?.off!=='function'||typeof runtime.setTimeout!=='function')return finish('unsupported-runtime');
  const now=()=>runtime.performance?.now?.()??Date.now();

  // Rendering or rendered: a refresh of what is already open. Older runtimes
  // without render states fall back to the rendered flag.
  const states=runtime.foundry?.applications?.api?.ApplicationV2?.RENDER_STATES;
  const open=app=>states&&typeof app.state==='number'?app.state===states.RENDERING||app.state===states.RENDERED:!!app.rendered;

  function adopt(app){
    if(adopted.has(app))return finish('installed',{adopted:true});
    const {render,close,_replaceHTML:replace,_onRender:onRender}=app;
    if(typeof render!=='function'||typeof close!=='function'||typeof replace!=='function'||typeof onRender!=='function'
      ||['render','close','_replaceHTML','_onRender'].some(name=>Object.hasOwn(app,name))||!replacesContent(replace))
      return finish('unsupported-source',{detail:'TrackerPF2eHUD no longer renders the way this patch expects.'});

    // Looked up on each call, so wrappers registered on the prototype chain after
    // the tracker was adopted still run.
    const prototype=Object.getPrototypeOf(app),adoptedMethods={render,close,_replaceHTML:replace,_onRender:onRender};
    const inherited=name=>{const fn=Reflect.get(prototype,name,app);return typeof fn==='function'?fn:adoptedMethods[name];};

    // Render and close share the application's queue, so a close can still be
    // waiting behind a slow render when the window ends. Each close starts a new
    // generation, and a refresh merged in an earlier one is dropped.
    let last=-Infinity,pending=null,drop=null,latest,generation=0;
    app.render=function(options,_options){
      if(this!==app||_options!==undefined||!plain(options)||!open(app))return inherited('render').call(this,options,_options);
      const time=now();
      if(!pending&&time-last>=windowMs){last=time;stats.rendered++;return inherited('render').call(app,options);}
      stats.throttled++;latest=options;
      if(!pending){
        const scheduled=generation;
        pending=new Promise((resolve,reject)=>{
          drop=()=>resolve(app);
          runtime.setTimeout(()=>{
            if(scheduled!==generation)return;
            pending=drop=null;last=now();
            stats.rendered++;
            Promise.resolve(inherited('render').call(app,latest)).then(resolve,reject);
          },Math.max(0,windowMs-(time-last)));
        });
      }
      return pending;
    };
    app.close=function(...args){
      if(this===app){
        generation++;last=-Infinity;
        if(drop){stats.dropped++;drop();}
        pending=drop=null;
      }
      return inherited('close').apply(this,args);
    };

    let content=null,html=null,collapsed,unchanged=false;
    app._replaceHTML=function(result,element,options){
      unchanged=typeof result==='string'&&element===content&&!!element?.isConnected&&result===html&&options?.collapsed===collapsed;
      if(unchanged){stats.identical++;return;}
      inherited('_replaceHTML').call(this,result,element,options);
      content=element;html=result;collapsed=options?.collapsed;
    };
    // Effects panel height, scroll-to-current and alternate controls only
    // restate what the unchanged DOM already shows.
    app._onRender=function(context,options){
      if(unchanged){unchanged=false;return Promise.resolve();}
      return inherited('_onRender').call(this,context,options);
    };
    adopted.add(app);
    return finish('installed',{adopted:true});
  }

  const current=runtime.foundry?.applications?.instances?.get?.(ID);
  if(current)return adopt(current);
  const hook=Hooks.on('renderApplicationV2',app=>{
    if(app?.id!==ID)return;
    Hooks.off('renderApplicationV2',hook);
    adopt(app);
  });
  return finish('installed',{adopted:false});
}
