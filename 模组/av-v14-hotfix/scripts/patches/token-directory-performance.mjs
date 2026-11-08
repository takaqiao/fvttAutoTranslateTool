import {sourceTokens} from '../source-contract.mjs';

const TARGET='foundry.applications.sidebar.tabs.PlaceableDirectory.prototype._canRender';
const installed=new WeakSet();

// These are the native render consumers bypassed by a coordinate-only update.
// Compare tokens, not formatting or the core version label.
const SOURCES={
  "ApplicationV2.render": `async render(options={}, _options={}) {
    if ( typeof options === "boolean" ) options = Object.assign(_options, {force: options});
    return this.#semaphore.add(this.#render.bind(this), options);
  }`,
  "AbstractSidebarTab.render": `async render(options, _options) {
    if ( this.popout?.rendered ) this.popout.render(options, _options);
    return super.render(options, _options);
  }`,
  "AbstractSidebarTab.isPopout": `get isPopout() {
    return this.options.window.frame;
  }`,
  "PlaceableDirectory.tab": `get tab() {
    return this.#tab;
  }`,
  "PlaceableDirectory._canRender": `_canRender(options) {
    if ( game.settings.get("core", "noCanvas") ) return false;
    const { renderContext } = options;
    if ( !renderContext ) return true;
    const layerName = this.tabGroups.sheet;
    if ( !layerName ) return true;
    const layerClass = CONFIG.Canvas.layers[layerName].layerClass;
    const collection = foundry.utils.getDocumentClass(layerClass.documentName).metadata.collection;
    return [\`create\${collection}\`, \`update\${collection}\`, \`delete\${collection}\`].includes(renderContext);
  }`,
  "PlaceableDirectory._onRender": `async _onRender(context, options) {
    await super._onRender(context, options);
    await this._renderTab(context, options);
    if ( canvas.scene ) canvas.scene.apps[this.id] = this;
  }`,
  "PlaceableDirectory._prepareContext": `async _prepareContext(options) {
    await canvas.initializing;
    const activeLayer = canvas.activeLayer?.options.name;
    const layerClass = CONFIG.Canvas.layers[activeLayer]?.layerClass;
    if ( CONFIG[layerClass?.documentName]?.sidebar ) this.tabGroups.sheet = activeLayer;
    this.tabGroups.sheet ??= "tokens";
    return super._prepareContext(options);
  }`,
  "PlaceableDirectory._renderTab": `async _renderTab(context, options) {
    this.#tab?._onDeactivateTab();
    const tab = this.tabGroups.sheet;
    let app = this.#tabs[tab];
    if ( !app ) {
      const { layerClass } = CONFIG.Canvas.layers[tab] ?? {};
      const { sidebar } = CONFIG[layerClass?.documentName] ?? {};
      if ( !sidebar ) return;
      const collectionName = foundry.documents.Scene.getCollectionName(layerClass.documentName);
      app = new sidebar.applicationClass({ collectionName, directory: this });
      this.#tabs[tab] = app;
    }
    this.#tab = app;
    await app.render({ force: true });
    app.element.dataset.applicationPart = "tab";
    const prior = this.element.querySelector("[data-application-part=tab]");
    if ( app.element !== prior ) prior.replaceWith(app.element);
  }`,
  "PlaceableTab.collectionName": `get collectionName() {
    return this.options.collectionName;
  }`,
  "PlaceableTab.layer": `get layer() {
    return canvas?.getCollectionLayer(this.collectionName);
  }`,
  "PlaceableTab._getEntryLabel": `_getEntryLabel(entry) {
    return entry.name || entry.id;
  }`,
  "PlaceableTab._prepareDirectoryContext": `async _prepareDirectoryContext(context, options) {
    const controlled = Object.fromEntries(this.layer.controlled.map(o => [o.id, o]));
    context.directoryPartial = this.constructor.DIRECTORY_PARTIAL;
    context.entryPartial = this.constructor.ENTRY_PARTIAL;
    const promises = [];
    for ( const entry of canvas.scene?.[this.collectionName] ?? [] ) {
      if ( !entry.visible ) continue;
      const { id } = entry;
      const css = id in controlled ? "active" : "";
      const ctx = { css, id, label: this._getEntryLabel(entry) };
      promises.push(this._prepareEntry(entry, ctx));
    }
    context.entries = await Promise.all(promises);
    context.entries.sort((a, b) => a.label.localeCompare(b.label, game.i18n.lang));
    return context;
  }`,
  "PlaceableTab._prepareEntry": `async _prepareEntry(entry, context) {
    context.hidden = entry.hidden;
    context.locked = entry.locked;
    return context;
  }`,
  "PlaceableTab._onDeactivateTab": `_onDeactivateTab() {
    this._filterState.elevation = { bottom: -Infinity, top: Infinity };
    this._filterState.visible.clear();
    this._filterState.dialog?.close();
    // Preserve filtered levels when moving between tabs.
  }`,
  "PlaceableTab._onRender": `async _onRender(context, options) {
    await super._onRender(context, options);
    new DragDrop.implementation({
      dragSelector: "[data-entry-id]",
      permissions: {
        dragstart: () => true,
        drop: () => false
      },
      callbacks: {
        dragstart: this._onDragStart.bind(this)
      }
    }).bind(this.element);
    this._searchFilter.bind(this.element);
    for ( const li of this.element.querySelectorAll("[data-entry-id]") ) {
      li.addEventListener("pointerenter", this._onHoverInEntry.bind(this), { passive: true });
      li.addEventListener("pointerleave", this._onHoverOutEntry.bind(this), { passive: true });
    }
    this._applyFilters();
  }`,
  "PlaceableTab._matchesFilter": `_matchesFilter(entry) {
    return true;
  }`,
  "PlaceableTab._hasAdvancedFilters": `_hasAdvancedFilters() {
    const { elevation, hidden, locked } = this._filterState;
    return Number.isFinite(elevation.bottom) || Number.isFinite(elevation.top) || (hidden !== null)
      || (locked !== null);
  }`,
  "TokenTab._prepareDirectoryContext": `async _prepareDirectoryContext(context, options) {
    context = await super._prepareDirectoryContext(context, options);
    if ( (canvas.scene?.levels.size ?? 0) <= 1 ) return context;
    const noLevel = [];
    const groups = {};
    for ( const entry of context.entries ) {
      const level = canvas.scene.levels.get(entry.levelId);
      if ( level ) {
        const group = groups[level.id] ??= { id: level.id, label: level.name, entries: [] };
        group.entries.push(entry);
      }
      else noLevel.push(entry);
    }
    context.groups = canvas.scene.levels.reverseSorted.reduce((arr, l) => {
      const group = groups[l.id];
      if ( group?.entries.length ) arr.push(group);
      return arr;
    }, []);
    if ( noLevel.length ) context.groups.push({
      label: _loc("SIDEBAR.PLACEABLES.Categories.NoLevel"), entries: noLevel
    });
    return context;
  }`,
  "TokenTab._prepareEntry": `async _prepareEntry(entry, context) {
    context = await super._prepareEntry(entry, context);
    context.levelId = entry.level;
    return context;
  }`,
  "PlaceableTab._preparePartContext": `async _preparePartContext(partId, context, options) {
    context = await super._preparePartContext(partId, context, options);
    switch ( partId ) {
      case "directory": return this._prepareDirectoryContext(context, options);
      case "search": return this._prepareSearchContext(context, options);
    }
    return context;
  }`,
  "PlaceableTab._configureRenderParts": `_configureRenderParts(options) {
    const parts = super._configureRenderParts(options);
    parts.directory.templates.push(this.constructor.DIRECTORY_PARTIAL, this.constructor.ENTRY_PARTIAL);
    return parts;
  }`,
};

// Checked for what the skipped render needs instead of by text. Core 14.370
// added null-scene guards to all three without changing that.
const CAPABILITIES={
  // The skip calls this in place of a rebuild, so it must still recompute the
  // hidden state of every existing entry element.
  "PlaceableTab._applyFilters":source=>source.some(token=>token.kind==='literal'&&token.value.slice(1,-1)==='[data-entry-id]')
    &&source.some((token,index)=>token.value==='.'&&source[index+1]?.value==='hidden'&&source[index+2]?.value==='='),
  // Resolves those elements for _applyFilters.
  "PlaceableTab._getPlaceableFromElement":()=>true,
  // The skip leaves the search part as last rendered.
  "PlaceableTab._prepareSearchContext":()=>true
};

function parse(source){
  // The shared tokenizer predates private fields. Reserve a spelling that
  // cannot collide with an ordinary identifier in either compared source.
  if(source.includes('__avPrivate_'))throw new SyntaxError('Reserved source identifier');
  return sourceTokens(source.replaceAll('#','__avPrivate_'));
}
const tokens=source=>JSON.stringify(parse(source).map(token=>token.value));
const expected=new Map(Object.entries(SOURCES).map(([name,source])=>[name,tokens(source)]));
const directoryMethods={render:'AbstractSidebarTab',_prepareContext:'PlaceableDirectory',_onRender:'PlaceableDirectory',
  _renderTab:'PlaceableDirectory'};
const tabMethods={render:'ApplicationV2',_prepareDirectoryContext:'TokenTab',_prepareEntry:'TokenTab',
  _getEntryLabel:'PlaceableTab',_onRender:'PlaceableTab',_onDeactivateTab:'PlaceableTab',_applyFilters:'PlaceableTab',_matchesFilter:'PlaceableTab',
  _hasAdvancedFilters:'PlaceableTab',_getPlaceableFromElement:'PlaceableTab',_preparePartContext:'PlaceableTab',
  _prepareSearchContext:'PlaceableTab',_configureRenderParts:'PlaceableTab'};

function member(classes,key){
  const [owner,name]=key.split('.'),prototype=classes[owner]?.prototype;
  const descriptor=prototype&&Object.getOwnPropertyDescriptor(prototype,name);
  const fn=descriptor?.value??descriptor?.get;
  return typeof fn==='function'?{prototype,name,fn,getter:!!descriptor.get}:null;
}
const sourceOf=fn=>Reflect.apply(Function.prototype.toString,fn,[]);

function contracts(classes){
  const records=[];
  for(const [key,source] of expected){
    const record=member(classes,key);
    if(!record||tokens(sourceOf(record.fn))!==source)return null;
    records.push(record);
  }
  for(const [key,accepts] of Object.entries(CAPABILITIES)){
    const record=member(classes,key);
    if(!record||record.getter||!accepts(parse(sourceOf(record.fn))))return null;
    records.push(record);
  }
  return records;
}

function coordinates(changes){
  return Array.isArray(changes)&&changes.length>0&&changes.every(change=>{
    if(!change||typeof change!=='object'||Array.isArray(change)||typeof change._id!=='string'||!change._id)return false;
    const keys=Reflect.ownKeys(change);
    return keys.some(key=>key==='x'||key==='y')
      &&keys.every(key=>key==='_id'||key==='_stats'||((key==='x'||key==='y')&&Number.isFinite(change[key])));
  });
}

function visibleEntriesMatch(scene,tab,changes){
  const visible=new Set(),ids=new Set();
  for(const token of scene.tokens){
    ids.add(token.id);
    if(token.visible)visible.add(token.id);
  }
  if(changes.some(change=>!ids.has(change._id)))return false;
  const entries=tab.element.querySelectorAll('[data-entry-id]');
  if(entries.length!==visible.size)return false;
  for(const entry of entries)if(!visible.delete(entry.dataset.entryId))return false;
  return visible.size===0;
}

/** Keep the native filter update while avoiding unchanged Token directory HTML. */
export function installTokenDirectoryPerformancePatch({moduleId='av-v14-hotfix',runtime=globalThis,report=()=>{}}={}){
  const stats={skipped:0,delegated:0};
  const finish=status=>{const result={feature:'tokenDirectory',status,stats};report(result);return result;};
  const tabs=runtime.foundry?.applications?.sidebar?.tabs;
  const classes={...tabs,ApplicationV2:runtime.foundry?.applications?.api?.ApplicationV2,
    AbstractSidebarTab:runtime.foundry?.applications?.sidebar?.AbstractSidebarTab};
  const {PlaceableDirectory,PlaceableTab,TokenTab,AbstractSidebarTab}=classes;
  if(typeof runtime.libWrapper?.register!=='function'||!PlaceableDirectory?.prototype||!PlaceableTab?.prototype
    ||!TokenTab?.prototype||!AbstractSidebarTab?.prototype)return finish('unsupported-runtime');
  if(installed.has(PlaceableDirectory.prototype))return finish('already-installed');
  let records;
  try{records=contracts(classes);}catch{return finish('unsupported-source');}
  if(!records)return finish('unsupported-source');
  const directoryConsumers=Object.entries(directoryMethods).map(([name,owner])=>[name,classes[owner].prototype[name]]);
  const tabConsumers=Object.entries(tabMethods).map(([name,owner])=>[name,classes[owner].prototype[name]]);
  if(directoryConsumers.some(([name,fn])=>PlaceableDirectory.prototype[name]!==fn)
    ||tabConsumers.some(([name,fn])=>TokenTab.prototype[name]!==fn))return finish('unsupported-source');
  const scenes=new WeakMap();

  function canSkip(directory,options,sameScene){
    if(!sameScene||options.renderContext!=='updatetokens'||options.force||options.isFirstRender
      ||Reflect.ownKeys(options).some(key=>!['renderContext','renderData','force','isFirstRender'].includes(key))
      ||!coordinates(options.renderData))return false;
    const {canvas,CONFIG,ui}=runtime,scene=canvas?.scene,tab=directory.tab;
    if(!canvas?.ready||!scene||directory!==ui?.placeables||directory.isPopout||directory.popout?.rendered
      ||Object.getPrototypeOf(directory)!==PlaceableDirectory.prototype||Object.getPrototypeOf(tab)!==TokenTab.prototype
      ||CONFIG?.ui?.placeables!==PlaceableDirectory||CONFIG?.Token?.sidebar?.applicationClass!==TokenTab
      ||!directory.rendered||!tab.rendered||!directory.element?.isConnected||!tab.element?.isConnected
      ||scene.apps?.[directory.id]!==directory||canvas.activeLayer?.options.name!=='tokens'
      ||directory.tabGroups?.sheet!=='tokens'||tab.collectionName!=='tokens'||tab.layer!==canvas.tokens
      ||tab.options.directory!==directory)return false;
    // Native _renderTab resets height filters and closes their dialog before rebuilding.
    const {elevation,dialog}=tab._filterState;
    if(elevation?.bottom!==-Infinity||elevation?.top!==Infinity||dialog)return false;
    if(directoryConsumers.some(([name,fn])=>directory[name]!==fn)||tabConsumers.some(([name,fn])=>tab[name]!==fn))return false;
    for(const {prototype,name,fn,getter} of records){
      if(name==='_canRender')continue; // libWrapper owns this descriptor after registration.
      const descriptor=Object.getOwnPropertyDescriptor(prototype,name);
      if((getter?descriptor?.get:descriptor?.value)!==fn)return false;
    }
    return visibleEntriesMatch(scene,tab,options.renderData);
  }

  runtime.libWrapper.register(moduleId,TARGET,function(wrapped,options){
    // WRAPPER must call the next consumer, including its veto or exception.
    const result=wrapped(options);
    let skip=false;
    if(result===true){
      try{
        const scene=runtime.canvas?.scene,sameScene=scenes.has(this)&&scenes.get(this)===scene;
        scenes.set(this,scene);
        skip=canSkip(this,options,sameScene);
      }catch{/* An unreadable directory continues through its ordinary render. */}
    }
    if(!skip){stats.delegated++;return result;}
    // Filters also release newly hidden selections and refresh canvas flags.
    this.tab._applyFilters();
    stats.skipped++;
    return false;
  },'WRAPPER');
  installed.add(PlaceableDirectory.prototype);
  return finish('installed');
}
