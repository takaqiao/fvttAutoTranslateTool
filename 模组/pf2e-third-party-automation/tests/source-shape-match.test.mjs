import test from 'node:test';
import assert from 'node:assert/strict';
import {createShapeBinding} from '../scripts/source-shape-match.mjs';

const matches=(reference,observed,options)=>createShapeBinding(options).match(reference,observed,options)===observed.length;
const refused=(reference,observed,options)=>assert.throws(()=>createShapeBinding(options).match(reference,observed,options),{message:'source-shape-mismatch'});

test('a consistent renaming of short names matches and renders the reference in the observed names',()=>{
 const shapes=createShapeBinding(),observed='function g(x,y){let z=x+y;return z*x}';
 assert.equal(shapes.match('function f(a,b){let c=a+b;return c*a}',observed),observed.length);
 assert.equal(shapes.render('f(b,a,c)'),'g(y,x,z)');
});

test('matching starts at the given offset and returns where the region ends',()=>{
 const shapes=createShapeBinding(),observed='let q=1;f(x);g(x)';
 assert.equal(shapes.match('f(a)',observed,{at:8}),12);
 assert.equal(shapes.match(';g(a)',observed,{at:12}),observed.length);
});

for(const [name,reference,observed]of [
 ['one name given two names','a+a','x+y'],
 ['two names given one name','a+b','x+x'],
 ['exchanged arguments','f(a,b){g(a,b)}','f(x,y){g(y,x)}'],
 ['a removed statement','{a();b();c()}','{x();z()}'],
 ['an added statement','{a();c()}','{x();y();z()}'],
 ['a shorter region','a();b()','x()'],
])test(`${name} is refused`,()=>refused(reference,observed));

test('a refusal names the first differing token and its offset',()=>{
 assert.throws(()=>createShapeBinding().match('e.update(n)','q.delete(n)'),{message:'source-shape-mismatch',index:2,expected:'update',found:'delete',offset:2});
});

test('the renaming is shared by every region matched through one binding',()=>{
 const shapes=createShapeBinding();
 shapes.match('f(a)','g(x)');
 // A bound name is reported as the observed name it stands for.
 assert.throws(()=>shapes.match('a.id','y.id'),{message:'source-shape-mismatch',expected:'x',found:'y'});
 assert.throws(()=>shapes.match('b.id','x.id'),{message:'source-shape-mismatch',expected:'b',found:'x'});
 // A refused region binds nothing: b is still free to take y.
 assert.throws(()=>shapes.match('b+a','y+y'),{message:'source-shape-mismatch'});
 assert.equal(shapes.match('a.id+b','x.id+y'),6);
});

for(const [name,reference,observed,same]of [
 ['a property name','e.id','q.id','q.di'],
 ['an optional property name','e?.id','q?.id','q?.di'],
 ['an object key','f({id:e})','g({id:q})','g({di:q})'],
 ['a short object key','f({a:e,b:n})','g({a:q,b:r})','g({c:q,b:r})'],
 ['a shorthand property','f({id,e})','g({id,e})','g({id,q})'],
 ['an object method name','f({at(e){return e}})','g({at(q){return q}})','g({to(q){return q}})'],
 ['an accessor name','f({get id(){return e}})','g({get id(){return q}})','g({get di(){return q}})'],
 ['a destructured key','let{id:e,...n}=a','let{id:q,...r}=s','let{di:q,...r}=s'],
 ['a class member name','class{at(e){return e}static of=e;#p=1}','class{at(q){return q}static of=q;#w=1}','class{to(q){return q}static of=q;#w=1}'],
 ['a longer identifier','game.user(e)','game.user(q)','gamo.user(q)'],
 ['a keyword','for(let e of n)e()','for(let q of r)q()','for(let q in r)q()'],
 ['a string','f(e,"a b")','g(q,"a b")','g(q,"q b")'],
 ['template text','f(`a ${e.id} b`)','g(`a ${q.id} b`)','g(`q ${q.id} b`)'],
 ['a regular expression','f(e,/a\\/[/b]/g)','g(q,/a\\/[/b]/g)','g(q,/q\\/[/b]/g)'],
 ['a number','e+1e3','q+1e3','q+1e4'],
 ['an operator','e&&n','q&&r','q||r'],
 ['whitespace','let e=1;return e','let q=1;return q','let q=1; return q'],
 ['a comment','e/*a*/+n','q/*a*/+r','q/*b*/+r'],
])test(`${name} is compared verbatim around renamed names`,()=>{
 assert.equal(matches(reference,observed),true);refused(reference,same);
});

test('names inside a template substitution are renamed and the template text is not',()=>{
 const shapes=createShapeBinding();
 shapes.match('f(`module.${A.id}/${t}`,t)','g(`module.${M.id}/${e}`,e)');
 assert.equal(shapes.render('`${A}${t}A t`'),'`${M}${e}A t`');
});

test('a slash after a value divides and the names around it are renamed',()=>{
 assert.equal(matches('a/b/c+(a)/b+a[0]/b','x/y/z+(x)/y+x[0]/y'),true);
 assert.equal(matches('if(a)/b/.test(c)','if(x)/b/.test(z)'),true);
 refused('if(a)/b/.test(c)','if(x)/y/.test(z)');
 // A property may be named like a keyword and is still a value.
 assert.equal(matches('a.in/b/c','x.in/y/z'),true);
 refused('f(b){return b.in/b/2}','f(y){return y.in/b/2}');
 refused('f(b){return b.if(b)/b/2}','f(y){return y.if(y)/b/2}');
});

test('private names are renamed in a namespace of their own',()=>{
 const shapes=createShapeBinding();
 shapes.match('#a(a){return this.#a(a)+this.#b}','#x(x){return this.#x(x)+this.#y}',{within:'class'});
 assert.equal(shapes.render('this.#b(a)'),'this.#y(x)');
 refused('this.#a+this.#b','this.#x+this.#x');refused('this.#a+this.#a','this.#x+this.#y');refused('this.#a','this.a');
});

test('host globals stay verbatim when the caller names them',()=>{
 assert.equal(matches('ui.error(e)','uj.error(q)'),true);
 refused('ui.error(e)','uj.error(q)',{fixed:['ui']});
 assert.equal(matches('ui.error(e)','ui.error(q)',{fixed:['ui']}),true);
 refused('new Map(e)','new Set(q)');
});

test('a fragment cut from a class keeps its leading member name verbatim',()=>{
 assert.equal(matches('at(e){return e}','to(q){return q}'),true);
 refused('at(e){return e}','to(q){return q}',{within:'class'});
 assert.equal(matches('at(e){return e}','at(q){return q}',{within:'class'}),true);
});

test('a name the reference adds keeps its own name unless an observed region uses it',()=>{
 const shapes=createShapeBinding();
 shapes.match('f(a,b)','x(y,z)');
 assert.equal(shapes.render('let x=f(a),s=b;return x+s'),'let w=x(y),s=z;return w+s');
 assert.equal(shapes.render('x'),'w');
 // The added names are bound like any other, so a later region cannot claim them.
 assert.throws(()=>shapes.match('c','w'),{message:'source-shape-mismatch'});
});

test('matching stops with the reference, so a region has to end at its own closing token',()=>{
 assert.equal(createShapeBinding().match('a();c()','x();y();z()'),7);
});

test('rendering an unmatched private name is refused',()=>{
 assert.throws(()=>createShapeBinding().render('this.#a'),{message:'source-shape-unbound',expected:'#a'});
});

for(const [name,reference]of [
 ['a slash after a closing brace','{}/a/g'],
 ['a slash after an increment','a++/b/c'],
 ['an unterminated string','f("a'],
 ['an unterminated template','f(`a${b}'],
 ['an unknown character','a@b'],
])test(`a reference holding ${name} is refused as unsupported`,()=>{
 assert.throws(()=>createShapeBinding().match(reference,reference),{message:'source-shape-unsupported'});
});

test('an observed region the scanner cannot classify does not match',()=>{
 assert.throws(()=>createShapeBinding().match('a;b()','x;y@'),{message:'source-shape-mismatch',found:null,offset:3});
});
