import vm from 'node:vm';
import {readFileSync} from 'node:fs';
import {nativeHooks} from './hook-harness.mjs';
import {timestampEnvironment} from './timestamps-harness.mjs';

const fixture=JSON.parse(readFileSync(new URL('./fixtures/calendaria-hud.json',import.meta.url),'utf8'));
export const TICK='calendaria.visualTick',WORLD='calendaria.worldTimeUpdated';

// Run the captured HUD, including its private methods. Layout and canvas drawing
// are external here; Text replacement counts are verified in the browser too.
export function calendariaEnvironment({source=fixture.HUD}={}){
  const {Node,Text,Element:BaseElement}=timestampEnvironment(),Hooks=nativeHooks();
  const state={time:'09:00',date:'Day 1',sky:0,timers:[],renders:[],format:null,attach:null};
  class Element extends BaseElement{
    constructor(className='',parent=null){
      super(className);this.tagName='SPAN';this.parentElement=parent;this.htmlWrites=0;
      this.classList={toggle(){},add(){},remove(){}};this.style={setProperty(){},removeProperty(){}};
    }
    addEventListener(){}
    getBoundingClientRect(){return {width:300,height:60,top:0,bottom:60,left:0};}
    set innerHTML(value){this.htmlWrites++;this.childNodes=[new Element('html')];this.html=value;}
    get innerHTML(){return this.html??this.textContent;}
  }
  function dom(){
    const root=new Element('calendaria-hud'),bar=new Element('bar'),time=new Element('time',new Element('time-wrapper')),date=new Element('date',new Element('info-panel'));
    time.append(new Text(state.time));date.append(new Text(state.date));
    root.nodes={'.time':time,'.date':date,'.bar':bar};root.querySelector=selector=>root.nodes[selector]??null;
    bar.querySelector=selector=>{
      if(selector===':scope > .section.left > .info-panel > span.date')return date;
      if(selector===':scope > .center > .time-wrapper > span.time')return time.parentElement.className==='time-wrapper'?time:null;
      if(selector===':scope > .center > .slice > .slice-overlay > span.time')return time.parentElement.className==='slice-overlay'?time:null;
      throw new Error('Unmodeled bar selector '+selector);
    };
    bar.querySelectorAll=()=>[];
    return {root,bar,time,date};
  }
  class Application{
    constructor(){this.element=dom().root;this.rendered=true;this.position={left:0,top:0};}
    _onRender(){}
    async _onClose(){}
    setPosition(position){Object.assign(this.position,position);}
    render(options){state.renders.push(options);}
  }
  const settings=new Map([['CALENDAR_HUD_POSITION',{left:0,top:0}]]);
  const game={modules:new Map([['calendaria',{active:true}]]),user:{isGM:true},settings:{get:(_id,key)=>settings.get(key)},time:{components:{year:1,month:0,dayOfMonth:1,hour:9}},scenes:{}};
  const constants=new Proxy({},{get:(_target,key)=>key});
  const calendar={days:{hoursPerDay:24}},clock={running:false,predictedWorldTime:0};
  const globals={gu:Base=>Base,Au:Application,o:{},e:{ID:'calendaria'},n:constants,E:{VISUAL_TICK:TICK,WORLD_TIME_UPDATED:WORLD},
    game,Hooks,Uc:{getActiveCalendar:()=>calendar},Do:clock,Oe:{getActiveZone:()=>null},Se:()=>false,
    hr:element=>state.attach?.(element),ks:()=>0,Ws:()=>null,Qs(){},Gs(){},Us(){},
    fn:(_calendar,_components,kind)=>{state.format?.(kind);return kind==='hudTime'?state.time:state.date;},
    en:value=>value.includes('<'),tn:value=>value,nn:value=>value.replace(/<[^>]*>/g,''),
    bs:value=>value,Ts:()=>{state.sky++;return [];},vs:()=>0,he:()=>false,
    foundry:{applications:{ux:{Draggable:{implementation:class{_onDragMouseDown(){}}}}}},
    window:{innerWidth:1200,innerHeight:800},setTimeout:fn=>{state.timers.push(fn);return state.timers.length;},clearTimeout(){}};
  const HUD=vm.runInNewContext(`(${source})`,globals),registrations=[];
  const libWrapper={
    register(moduleId,target,fn,type){
      const descriptor=Object.getOwnPropertyDescriptor(HUD.prototype,'_onRender');
      const data={package_info:{id:moduleId},fn,type};
      let chain=descriptor.get?._lib_wrapper,get=descriptor.get;
      if(!chain){
        chain={_wrapped:descriptor.value,records:[],listeners:[],get_fn_data(_setter,listener=false){return listener?this.listeners:this.records;}};
        function handler(...args){
          const fn=chain.records[0]?.fn;
          return fn?Reflect.apply(fn,this,[chain._wrapped.bind(this),...args]):Reflect.apply(chain._wrapped,this,args);
        }
        get=()=>handler;get._lib_wrapper=chain;
        Object.defineProperty(HUD.prototype,'_onRender',{get,set(value){chain._wrapped=value;},configurable:true});
      }
      chain.records.push(data);
      registrations.push({moduleId,target,fn,type,chain,descriptor,get,data});return registrations.length;
    },
    unregister(moduleId,id){
      const record=registrations[id-1];record.unregistered=true;
      // Current libWrapper leaves its accessor installed after the last removal.
      record.chain.records.splice(record.chain.records.indexOf(record.data),1);
    }
  };
  const runtime={Node,Hooks,game,libWrapper,CALENDARIA:{apps:{HUD}}};
  const hud=new HUD(),render=(options={isFirstRender:true})=>hud._onRender({},options);
  const tick=(...args)=>Hooks.events[TICK].at(-1).fn(...args),world=(...args)=>Hooks.events[WORLD].at(-1).fn(...args);
  return {runtime,HUD,Application,hud,Hooks,Node,Text,Element,dom,state,settings,game,clock,registrations,render,tick,world};
}
