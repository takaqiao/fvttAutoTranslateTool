import {TOOL_SHAPE} from '../scripts/native-source-shapes.mjs';

const WRAPPER='r("_preUpdate",this.#h)';
// What the bundle holds about the seam outside the regions: the hook wrapper and the helper registrations.
const REGISTRATIONS=['a(ea,"displayEmiting")','a(Ne,"userIsGM")','a(Ko,"isTokenObject")','a(Jx,"isValidTargetDocuments")'];
const HELPERS=`function ea(){}${REGISTRATIONS[0]};function Ne(){return!0}${REGISTRATIONS[1]};function Ko(t){return!1}${REGISTRATIONS[2]};function Jx(t){return!1}${REGISTRATIONS[3]};`;
// The reference regions where esbuild places them: the free functions at the top
// level of the bundle, the members inside the registered class, another class after it.
export const referenceToolbelt=`"use strict";(()=>{var a=(t,e)=>t;${HELPERS}${TOOL_SHAPE.socket};${TOOL_SHAPE.transport}${TOOL_SHAPE.conversion}var Tool=class{${TOOL_SHAPE.binding}${TOOL_SHAPE.validity}#v(e,n){return!1}#f(){}#c(){}init(){${WRAPPER}}${TOOL_SHAPE.receive}${TOOL_SHAPE.forward}},Next=class{static{a(this,"NextTool")}};})();\n`;

export function edited(source,before,after){
 if(source.split(before).length!==2)throw Error('toolbelt-shape-fixture-edit');
 return source.replace(before,()=>after);
}
/** `source` with one change made inside the reference region `key`. */
export const editedRegion=(source,key,before,after)=>edited(source,TOOL_SHAPE[key],edited(TOOL_SHAPE[key],before,after));

const MINIFIED='_e t e n i a o s c AA r ne ea UA Xo u Ne Gy Ky A MA RA zA Ko Jx l f d m p g h'.split(' '),PRIVATE='#t #g #e #f #n #c #h #v'.split(' ');
/** Every minified name of the reference regions, given another name of its kind:
 * $00, $01, ... which a bundle of this size does not reach, and #w0, #w1, ... */
export const otherNames=new Map([...MINIFIED.map((name,index)=>[name,'$'+index.toString(36).padStart(2,'0')]),...PRIVATE.map((name,index)=>[name,'#w'+index])]);
// Whole words that are not member accesses; no string in the regions holds one of these names as a word.
const member=(text,at)=>text[at-1]==='.'&&text[at-2]!=='.';
export const renamed=(text,names=otherNames)=>text.replace(/#?[A-Za-z_$][\w$]*/g,(word,at)=>word[0]!=='#'&&member(text,at)?word:names.get(word)??word);
/** A build holding the reference regions, with everything the seam reads from it renamed.
 * The rest of the bundle keeps its names, so the result is matched but never run. */
export function renamedSeams(source,names=otherNames){
 for(const text of [TOOL_SHAPE.socket,TOOL_SHAPE.transport,TOOL_SHAPE.conversion,TOOL_SHAPE.binding,TOOL_SHAPE.validity,TOOL_SHAPE.receive,TOOL_SHAPE.forward,WRAPPER,...REGISTRATIONS])source=edited(source,text,renamed(text,names));
 return source;
}
