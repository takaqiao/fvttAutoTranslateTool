/** Rename-equivalence for regions of minified JavaScript.
 *
 * Two regions have the same shape when their token streams are identical
 * except for a one-to-one renaming of the names a minifier assigns: short
 * identifiers in binding or reference position, and every private name.
 * Property names, object keys, member names, longer identifiers, literals,
 * punctuation, comments and whitespace are compared verbatim, so a change in
 * logic still changes the shape. Constructs the scanner cannot classify with
 * certainty are refused rather than guessed.
 *
 * Adapted from the source-shape matcher in pf2e-third-party-automation; only
 * the normalization half is used here, as in pf2e-compendium-extra-cn. No
 * dependencies; runs in Node and in the browser, synchronously. */
const KEYWORDS=new Set('as async await break case catch class const continue debugger default delete do else enum export extends false finally for from function get if import in instanceof let new null of return set static super switch this throw true try typeof var void while with yield'.split(' '));
// Short global names a minifier never assigns; callers add those of their host.
const GLOBALS=['NaN','Map','Set','URL','CSS'];
const VALUES=new Set(['this','super','null','true','false']),OPERATORS=new Set(['return','typeof','case','do','else','in','instanceof','new','delete','void','throw']);
const CONTROL=new Set(['if','while','for','with']),MODIFIERS=new Set(['get','set','async','static']);
const BLOCK_WORDS=new Set(['else','try','finally','do','static']),BLOCK_PUNCT=new Set([')','=>',';','{','}']);
const PUNCT=['>>>=','...','===','!==','**=','<<=','>>=','>>>','&&=','||=','??=','=>','==','!=','<=','>=','&&','||','??','?.','++','--','+=','-=','*=','%=','&=','|=','^=','<<','>>','**','{','}','(',')','[',']',';',',','<','>','+','-','*','%','&','|','^','!','~','?',':','=','.'];
const SPACE=/[ \t\r\n]+/y,COMMENT=/\/\/[^\r\n]*|\/\*[^]*?\*\//y,WORD=/[A-Za-z_$][\w$]*/y,PRIVATE=/#[A-Za-z_$][\w$]*/y,FLAGS=/[A-Za-z]*/y;
const NUMBER=/(?:0[xX][\da-fA-F_]+|0[oO][0-7_]+|0[bB][01_]+|(?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:[eE][+-]?\d[\d_]*)?)n?/y;
const SHORT=/^[A-Za-z_$][\w$]{0,2}$/;
const UNSUPPORTED=Symbol('unsupported');
const refuse=(reason,detail)=>{throw Object.assign(Error(reason),detail)};

/** Tokens of `source`, whitespace and comments included. Scanning stops at a
 * construct it cannot classify and records the offset in `unsupported`. */
function scan(source){
 const tokens=[],open=[];let i=0,last;
 const unsupported=()=>{throw UNSUPPORTED};
 const lex=pattern=>{pattern.lastIndex=i;return pattern.exec(source)?.[0].length??0};
 function quoted(){const quote=source[i];for(let j=i+1;j<source.length;j++){const c=source[j];if(c==='\\')j++;else if(c===quote)return j+1;else if(c==='\n'||c==='\r')break}return unsupported()}
 function template(from){for(let j=from;j<source.length;j++){const c=source[j];if(c==='\\')j++;else if(c==='`')return j+1;else if(c==='$'&&source[j+1]==='{'){open.push({template:true});return j+2}}return unsupported()}
 function regex(){let set=false;for(let j=i+1;j<source.length;j++){const c=source[j];if(c==='\\')j++;else if(c==='\n'||c==='\r')break;else if(c==='[')set=true;else if(c===']')set=false;else if(c==='/'&&!set){FLAGS.lastIndex=j+1;return j+1+FLAGS.exec(source)[0].length}}return unsupported()}
 // A slash divides after a value and starts a regular expression where a value is expected.
 function divides(){
  if(!last)return unsupported();
  if(last.kind==='word')return last.property||VALUES.has(last.text)||!KEYWORDS.has(last.text)||(OPERATORS.has(last.text)?false:unsupported());
  if(last.kind==='template')return last.text.endsWith('`');
  if(last.kind!=='punct')return true;
  if(last.text===')')return !last.control;
  return last.text===']'||(['}','++','--'].includes(last.text)?unsupported():false);
 }
 try{
  while(i<source.length){
   const c=source[i],next=source[i+1];let kind='punct',end,control=false;
   if(end=lex(SPACE)){kind='space';end+=i}
   else if(c==='/'&&(next==='/'||next==='*')){kind='space';end=i+(lex(COMMENT)||unsupported())}
   else if(c==='"'||c==="'"){kind='string';end=quoted()}
   else if(c==='`'){kind='template';end=template(i+1)}
   else if(c==='}'&&open.at(-1)?.template){open.pop();kind='template';end=template(i+1)}
   else if(end=lex(WORD)){kind='word';end+=i}
   else if(end=lex(PRIVATE)){kind='private';end+=i}
   else if(end=lex(NUMBER)){kind='number';end+=i;if(/[\w$]/.test(source[end]??''))unsupported()}
   else if(c==='/'){if(divides())end=i+(next==='='?2:1);else{kind='regex';end=regex()}}
   else{
    const text=c==='?'&&next==='.'&&/\d/.test(source[i+2]??'')?'?':PUNCT.find(p=>source.startsWith(p,i));
    if(!text)unsupported();end=i+text.length;
    if(text==='('||text==='['||text==='{')open.push({control:text==='('&&last?.kind==='word'&&CONTROL.has(last.text)&&!last.property});
    else if(text===')'||text===']'||text==='}')control=!!open.pop()?.control;
   }
   const property=kind==='word'&&last?.kind==='punct'&&(last.text==='.'||last.text==='?.');
   const token={kind,text:source.slice(i,end),control,property,role:0};
   if(kind!=='space')last=token;tokens.push(token);i=end;
  }
 }catch(error){if(error!==UNSUPPORTED)throw error;tokens.unsupported=i}
 return tokens;
}

/** Give each token its role: 0 verbatim, 1 renameable identifier, 2 renameable
 * private name. `within` names the construct a fragment is cut from ('class'
 * or 'object'), so that its member names stay verbatim. */
function classify(tokens,within,pinned){
 const frames=within?[{type:within,head:true}]:[],base=frames.length,code=tokens.filter(token=>token.kind!=='space');let classAt=-1;
 const is=(token,...texts)=>token?.kind==='punct'&&texts.includes(token.text);
 code.forEach((token,index)=>{
  const prev=code[index-1],next=code[index+1],frame=frames.at(-1),head=!!frame?.head,text=token.text;
  // `head` is set where a class or object literal expects a member name.
  if(frame)frame.head=false;
  if(token.kind==='private')token.role=2;
  else if(token.kind==='word'){
   if(text==='class'&&!token.property)classAt=frames.length;
   if(head)frame.head=MODIFIERS.has(text);
   else if(!token.property&&SHORT.test(text)&&!KEYWORDS.has(text)&&!pinned.has(text)&&!(is(prev,'{',',')&&is(next,':')))token.role=1;
  }else if(token.kind==='template'){
   if(text[0]==='}'&&frames.length>base)frames.pop();
   if(text.endsWith('${'))frames.push({type:'group'});
  }else if(token.kind==='punct'){
   if(text==='{'){
    const block=prev&&(prev.kind==='punct'?BLOCK_PUNCT.has(prev.text):prev.kind==='word'&&BLOCK_WORDS.has(prev.text));
    const type=classAt===frames.length?'class':block?'block':'object';
    if(type==='class')classAt=-1;frames.push({type,head:type!=='block'});
   }else if(text==='('||text==='[')frames.push({type:'group'});
   else if(text===')'||text===']'||text==='}'){
    if(frames.length>base)frames.pop();
    const outer=frames.at(-1);if(text==='}'&&outer?.type==='class')outer.head=true;
   }else if(text===','&&frame?.type==='object'||text===';'&&frame?.type==='class'||text==='*'&&head)frame.head=true;
  }
 });
 return tokens;
}

/** A stable token representation of `source`, for hashing a region without
 * storing it. Renameable names receive numbers in first-use order; all other
 * tokens keep their text. `fixed` lists short host globals that must stay
 * verbatim. Throws `source-shape-unsupported` for a region it cannot scan. */
export function normalizeSourceShape(source,{fixed=[],within}={}){
 const tokens=scan(source);
 if(tokens.unsupported!==undefined)refuse('source-shape-unsupported',{offset:tokens.unsupported});
 const names=[undefined,new Map(),new Map()];
 return JSON.stringify(classify(tokens,within,new Set([...GLOBALS,...fixed])).map(token=>{
  const table=names[token.role];
  if(!table)return [token.kind,token.text];
  if(!table.has(token.text))table.set(token.text,table.size);
  return [token.role,table.get(token.text)];
 }));
}
