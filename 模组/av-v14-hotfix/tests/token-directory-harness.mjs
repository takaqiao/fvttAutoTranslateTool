import fs from 'node:fs';
import vm from 'node:vm';

export const fixture=JSON.parse(fs.readFileSync(new URL('./fixtures/token-directory-native.json',import.meta.url),'utf8'));
const methods=owner=>Object.entries(fixture.methods).filter(([key])=>key.startsWith(`${owner}.`)).map(([,source])=>source).join('\n');

export function directoryEnvironment({gm=true,mutateSource=source=>source}={}){
  const registrations=[],events=[],elements=[],documents=[];
  const scene={id:'scene',tokens:documents,apps:{},levels:{size:1},getEmbeddedCollection:()=>({get:id=>documents.find(doc=>doc.id===id)})};
  const layer={options:{name:'tokens'},controlled:[],setAllRenderFlags:flags=>events.push(['flags',flags])};
  const canvas={ready:true,scene,tokens:layer,activeLayer:layer,getCollectionLayer:()=>layer,initializing:Promise.resolve()};
  const game={user:{isGM:gm},settings:{get:()=>false},i18n:{lang:'en'},activeTool:'select'};
  const CONFIG={Canvas:{layers:{tokens:{layerClass:{documentName:'Token'}}}},Token:{sidebar:{}},ui:{}};
  const foundry={utils:{getDocumentClass:()=>({metadata:{collection:'tokens'}})},data:{fields:{SchemaField:class {},NumberField:class {}}}};
  const group={meta:{textContent:''},querySelector:()=>group.meta,querySelectorAll:()=>elements};
  const element={isConnected:true,querySelectorAll:selector=>selector==='[data-entry-id]'?elements:selector==='.placeable-group'?[group]:[],querySelector:()=>null};
  const ui={placeablesPalette:{render:()=>events.push(['palette'])}};
  const context=vm.createContext({canvas,game,CONFIG,foundry,ui,
    DragDrop:{implementation:class {bind(){}}},
    _loc:key=>key,
    harnessRender:async(app,options)=>{
      if(typeof options==='boolean')options={force:options};
      options.isFirstRender=!app.rendered;
      if(app._canRender?.(options)===false)return app;
      if(app===ui.placeables){
        events.push(['directory-render']);
        await app._prepareContext(options);
        await app._onRender({},options);
      }else{
        const prepared=await app._prepareDirectoryContext({},options);
        elements.splice(0,elements.length,...prepared.entries.map(entry=>row(entry)));
        events.push(['tab-render',prepared]);
        await app._onRender({},options);
      }
      app.rendered=true;
      return app;
    }
  });
  vm.runInContext(mutateSource(`
    class ApplicationV2 {
      #semaphore={add:(fn,options)=>fn(options)};
      async #render(options){return harnessRender(this,options);}
      ${methods('ApplicationV2')}
      async _prepareContext(){return {};}
      async _onRender(){}
    }
    class AbstractSidebarTab extends ApplicationV2 {${methods('AbstractSidebarTab')}}
    class PlaceableTab extends ApplicationV2 {
      static #filterViewed=true;
      constructor(options){super();this.options=options;}
      ${methods('PlaceableTab')}
      _onDragStart(){}
      _onHoverInEntry(){}
      _onHoverOutEntry(){}
    }
    class TokenTab extends PlaceableTab {${methods('TokenTab')}}
    class PlaceableDirectory extends AbstractSidebarTab {
      #tab; #tabs={};
      ${methods('PlaceableDirectory')}
      setTab(tab){this.#tab=tab;this.#tabs.tokens=tab;}
      _updateFilterPip(){ }
    }
    this.classes={ApplicationV2,AbstractSidebarTab,PlaceableDirectory,PlaceableTab,TokenTab};
  `),context);
  const {classes}=context;
  const {AbstractSidebarTab,ApplicationV2,...tabs}=classes;
  foundry.applications={api:{ApplicationV2},sidebar:{AbstractSidebarTab,tabs}};
  CONFIG.Token.sidebar.applicationClass=classes.TokenTab;
  CONFIG.ui.placeables=classes.PlaceableDirectory;
  const directory=new classes.PlaceableDirectory();
  directory.options={window:{frame:false}};directory.id='placeables';directory.tabGroups={sheet:'tokens'};directory.rendered=true;
  const tab=new classes.TokenTab({collectionName:'tokens',directory});
  tab.rendered=true;tab.element=element;tab.schema={getField:()=>new foundry.data.fields.NumberField()};
  tab._filterState={elevation:{bottom:-Infinity,top:Infinity},levels:new Set(),hidden:null,locked:null,visible:new Set()};
  tab._searchFilter={constructor:{cleanQuery:s=>s},bind(){},rgx:null};
  directory.element={isConnected:true,querySelector:()=>element};
  element.dataset={};directory.setTab(tab);scene.apps[directory.id]=directory;ui.placeables=directory;
  const runtime={foundry,CONFIG,canvas,game,ui,libWrapper:{register(moduleId,target,fn,type){
    registrations.push({moduleId,target,fn,type});
    const original=classes.PlaceableDirectory.prototype._canRender;
    classes.PlaceableDirectory.prototype._canRender=function(...args){return fn.call(this,original.bind(this),...args);};
  }}};
  function row(entry){
    const classes=new Set(entry.css?[entry.css]:[]);
    return {dataset:{entryId:entry.id},label:entry.label,documentHidden:entry.hidden,locked:entry.locked,levelId:entry.levelId,
      hidden:false,classList:{remove:name=>classes.delete(name),contains:name=>classes.has(name)},
      closest(){return this;},addEventListener(){}};
  }
  function add(id,values={}){
    const doc={id,name:id,x:0,y:0,hidden:false,locked:false,elevation:0,level:'level',viewed:true,rendered:true,
      get visible(){return game.user.isGM||!this.hidden;},...values};
    doc.object={id,controlled:false,release:options=>{events.push(['release',id,options]);doc.object.controlled=false;}};
    documents.push(doc);elements.push(row({id,label:doc.name}));return doc;
  }
  const movement=(changes=[{_id:'a',x:4000}],extra={})=>({renderContext:'updatetokens',renderData:changes,force:false,...extra});
  return {runtime,classes,directory,tab,scene,canvas,game,CONFIG,events,elements,documents,registrations,group,add,movement,
    render:options=>directory.render(options??movement()),canRender:options=>directory._canRender(options??movement())};
}
