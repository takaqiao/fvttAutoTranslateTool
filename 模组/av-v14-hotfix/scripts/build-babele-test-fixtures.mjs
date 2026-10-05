/** Rebuild small local regression fixtures from explicitly supplied original source files. */
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {basename} from 'node:path';

const [corePath,babelePath,libWrapperPath]=process.argv.slice(2);
if(!corePath||!babelePath||!libWrapperPath)throw new Error('Usage: node scripts/build-babele-test-fixtures.mjs <foundry.mjs> <babele-ondemand-patch.js> <lib-wrapper.js>');
const sources=[corePath,babelePath,libWrapperPath].map(path=>{
  const bytes=readFileSync(path);
  return {path,name:basename(path),sha256:createHash('sha256').update(bytes).digest('hex'),text:bytes.toString('utf8').replaceAll('\r\n','\n')};
});
const [core,babele,lib]=sources;
if(core.sha256!=='02248043922e265f368a07922543acb54a7c6402b848ac77e9783093be66545e')throw new Error('Expected the verified Foundry 14.367 source capture.');
if(babele.sha256!=='21b1b6453a5c6c09dc99da330eb70a17ff80e7ff3790eae12f42be4d16336848')throw new Error('Expected the clean pf2e_compendium_chn 3.1.2 baseline.');
if(!lib.text.includes('const H="1.13.5.1"'))throw new Error('Expected libWrapper 1.13.5.1 source.');
const ranges=[];
function section(source,name,start,end){
  if(start<0||end<=start)throw new Error(`Cannot locate ${name} in ${source.name}`);
  const value=source.text.slice(start,end).trim();
  ranges.push({name,source:source.name,sourceSha256:source.sha256,startLine:source.text.slice(0,start).split('\n').length,
    endLine:source.text.slice(0,end).split('\n').length,bytes:Buffer.byteLength(value),sha256:createHash('sha256').update(value).digest('hex')});
  return value;
}
function topLevel(source,kind,name){
  const pattern=kind==='class'?new RegExp(`^class ${name}(?: |\\{)`,'m'):new RegExp(`^(?:async )?function ${name}\\(`,'m');
  const start=source.text.search(pattern),end=source.text.indexOf('\n}',start)+2;
  return section(source,name,start,end);
}
const coreFixture={version:'14.367'};
for(const name of ['StringTree','WordTree','DocumentIndex'])coreFixture[name]=topLevel(core,'class',name);
const methodStart=core.text.indexOf('  indexDocument(document) {');
coreFixture.compendiumIndexDocument=section(core,'CompendiumCollection.indexDocument',methodStart,core.text.indexOf('\n  /* -------------------------------------------- */',methodStart));
const babeleFixture={version:'3.1.2',functions:{}};
for(const name of ['scheduleDocumentIndexRebuild','rebuildDocumentIndexCompat','normalizePackId','translateIndexTitles','applyLightRuntimeTranslations']){
  babeleFixture.functions[name]=topLevel(babele,'function',name);
}
const wrapperStart=babele.text.lastIndexOf('\n\tlibWrapper.register(');
babeleFixture.wrapper=section(babele,'CompendiumCollection.indexDocument wrapper registration',wrapperStart,babele.text.lastIndexOf('\n}'));
const packageStart=lib.text.indexOf('function be(r)'),packageEnd=lib.text.indexOf('let ye=',packageStart);
const enumEnd=lib.text.indexOf(',o=function(t=!1)');
const libFixture={version:'1.13.5.1',packageCheck:section(lib,'package identity check',packageStart,packageEnd),
  enumFactory:section(lib,'enum factory',0,enumEnd)};
const directory=new URL('../tests/fixtures/',import.meta.url);
mkdirSync(directory,{recursive:true});
const outputs=[];
function write(name,value){
  const text=JSON.stringify(value,null,2)+'\n';
  writeFileSync(new URL(name,directory),text);
  outputs.push({name,bytes:Buffer.byteLength(text),sha256:createHash('sha256').update(text).digest('hex')});
}
write('babele-core-14.367.json',coreFixture);
write('babele-upstream-3.1.2.json',babeleFixture);
write('babele-lib-wrapper-1.13.5.1.json',libFixture);
write('babele-provenance.json',{format:1,generator:'scripts/build-babele-test-fixtures.mjs',
  sources:sources.map(({name,sha256})=>({name,sha256})),extraction:'Verbatim source excerpts; CRLF normalized to LF and outer whitespace trimmed. No function bodies rewritten.',ranges,outputs:[...outputs]});
process.stdout.write(JSON.stringify({files:outputs,totalBytes:outputs.reduce((total,file)=>total+file.bytes,0)},null,2)+'\n');
