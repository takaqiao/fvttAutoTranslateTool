// pf2e-item-activations turns each "<p><strong>Activate…" paragraph of an item description into an
// action. It names the action "<Activation>: " plus the paragraph's bold text with one "—" removed, and
// takes traits from the paragraph's first half-width "(…)". The Chinese compendium writes "启动——释放龙息"
// and "（专注，操作）", so the action becomes "启动项: —释放龙息" with no traits of its own (a half-width
// group of Chinese labels yields strings PF2e then drops as invalid traits).
//
// Corrected here, on actions the module is about to create: the extra leading dashes after "<Activation>: ",
// and, when the action has no traits besides the module's own, the labels in the paragraph's first
// full-width group (or half-width group holding Chinese) mapped back to trait slugs by their localized name.
// Labels without exactly one matching trait are dropped. Actions created earlier are left as they are.
//
// The module finds its actions by `flags["pf2e-item-activations"].grantedBy._id` and skips creating one
// whose `system.slug` an existing item already has. Neither is changed here: the slug stays the one the
// module derived from the uncorrected name, which sluggify makes the same as the corrected name's, so an
// action from before this correction and one from after are still the same action to the module.

const MODULE='pf2e-item-activations';
// Traits the module adds to every generated action, not read from the description.
const MODULE_TRAITS=new Set(['activation','invested']);
const HAN=/\p{Script=Han}/u;
const LEADING_DASHES=/^[—―]+\s*/;
const installed=new WeakSet();
const escapeRegExp=text=>text.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');

// The activation paragraph the module copied into the action, without its closing tag.
export function activationParagraph(description,activateText){
  const start=description.indexOf(`<p><strong>${activateText}`);
  if(start<0)return null;
  const end=description.indexOf('</p>',start);
  return description.slice(start,end<0?undefined:end);
}

// Labels from the first bracket group after the bold name. The name itself may hold an English gloss
// in full-width brackets ("燃火（Stoke Flames）"), so it is skipped.
export function traitLabels(paragraph){
  const nameEnd=paragraph.indexOf('</strong>');
  const rest=nameEnd<0?paragraph:paragraph.slice(nameEnd+'</strong>'.length);
  const group=/（([^）]*)）|\(([^)]*)\)/.exec(rest);
  if(!group)return [];
  const text=(group[1]??group[2]).replace(/<[^>]*>/g,'');
  // A half-width group of English trait slugs is what the module already parses.
  if(group[1]===undefined&&!HAN.test(text))return [];
  return text.split(/[，,、]/).map(label=>label.trim()).filter(Boolean);
}

// Localized trait name -> slug; a name shared by two traits maps to null.
export function traitSlugsByLabel(actionTraits,localize){
  const slugs=new Map();
  for(const [slug,key] of Object.entries(actionTraits)){
    const label=String(localize(key)).trim();
    if(!label)continue;
    slugs.set(label,slugs.has(label)&&slugs.get(label)!==slug?null:slug);
  }
  return slugs;
}

export function correctActivation(source,{activateText,activationText,actionTraits,localize}){
  const result={changes:{},renamed:false,filled:0,dropped:0};
  const description=source?.system?.description?.value;
  if(typeof description!=='string')return result;
  const paragraph=activationParagraph(description,activateText);
  if(!paragraph)return result;

  // The module's "[X] " marker for inactive actions may precede the name.
  const name=typeof source.name==='string'?source.name:'';
  const prefix=new RegExp(`^(?:\\[X\\] )?${escapeRegExp(activationText)}: `).exec(name);
  if(prefix){
    const rest=name.slice(prefix[0].length),trimmed=rest.replace(LEADING_DASHES,'');
    if(trimmed!==rest&&trimmed){result.changes.name=prefix[0]+trimmed;result.renamed=true;}
  }

  const traits=Array.isArray(source.system.traits?.value)?source.system.traits.value:null;
  if(traits&&actionTraits&&typeof actionTraits==='object'&&!traits.some(trait=>!MODULE_TRAITS.has(trait))){
    const labels=traitLabels(paragraph);
    if(labels.length){
      const slugs=traitSlugsByLabel(actionTraits,localize),found=[];
      for(const label of labels){
        const slug=slugs.get(label);
        if(slug)found.push(slug);else result.dropped++;
      }
      if(found.length){
        // Sorted, as ItemPF2e#_preCreate has already left them before this hook.
        result.changes.system={traits:{value:[...new Set([...found,...traits])].sort()}};
        result.filled=found.length;
      }
    }
  }
  return result;
}

export function installItemActivationsCnPatch({runtime=globalThis,report=()=>{}}={}){
  const finish=(status,extra)=>{const result={feature:'itemActivationsCn',status,...extra};report(result);return result;};
  if(runtime.game?.system?.id!=='pf2e')return finish('unsupported-system');
  if(!runtime.game.modules?.get?.(MODULE)?.active)return finish('skipped',{reason:'module-inactive'});
  const Hooks=runtime.Hooks,i18n=runtime.game.i18n;
  if(typeof Hooks?.on!=='function'||typeof i18n?.localize!=='function')return finish('unsupported-runtime');
  if(installed.has(Hooks))return finish('already-installed');
  // `renamed` / `traitsFilled` count corrected actions; `labelsDropped` counts labels without a unique trait.
  const stats={renamed:0,traitsFilled:0,labelsDropped:0};
  let failed=false;
  const hook=Hooks.on('preCreateItem',item=>{
    try{
      const source=item?._source;
      // Every action the module creates carries the granting item in this flag.
      const granted=source?.flags?.[MODULE]?.grantedBy;
      if(source?.type!=='action'||!granted||typeof granted!=='object')return;
      const localize=key=>i18n.localize(key);
      const {changes,renamed,filled,dropped}=correctActivation(source,{
        activateText:localize(`${MODULE}.code.activate`),activationText:localize(`${MODULE}.code.activation`),
        actionTraits:runtime.CONFIG?.PF2E?.actionTraits,localize});
      stats.labelsDropped+=dropped;
      if(!renamed&&!filled)return;
      item.updateSource(changes);
      if(renamed)stats.renamed++;
      if(filled)stats.traitsFilled++;
    }catch(error){
      // The action is still created as the module built it.
      if(!failed){failed=true;report({feature:'itemActivationsCn',status:'failed',detail:String(error),hook,stats});}
    }
  });
  installed.add(Hooks);
  return finish('installed',{hook,stats});
}
