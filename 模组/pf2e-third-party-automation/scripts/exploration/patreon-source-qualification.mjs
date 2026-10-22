import {canonicalJSON} from './revision-codec.mjs';

export const PATREON_SOURCE_SEAMS=Object.freeze({
 time:'8c32f064ac33ab4a0ce264c8d74c8acbf8621627f1a8ef6f111e9ecb606cfa92',
 manualImmunity:'bdc211e3b7e437e91f8d9ab2c8f54c6b189abd8789278989de3a5a5b84e566a0',
 createItem:'14162e56764de7549e40aa9dad55249c14fe072acf181045f288bd02bd9bd7df'
});
export const PATREON_SOURCE_SHAPES=Object.freeze({
 time:'f33480bbf2e8f7674d6d7bd6a4fafa2d3813ce09ec2e226dafee03a917cb7466',
 manualImmunity:'79915211a7bde205e12b5fb6a8456c7601196771ce86f060733af27fbc899021',
 createItem:'241f13cd4cc502cb8e880ee9a2f6ffca6b884249db424333bf9b0d6327ab3072'
});

export function isPatreonSourceQualified(value){
 try{
  const json=canonicalJSON(value),descriptor=JSON.parse(json);
  if(Reflect.ownKeys(value).length!==8||Reflect.ownKeys(value.seams).length!==3
   ||!['providerVersion','pf2eVersion'].every(key=>typeof descriptor[key]==='string'&&descriptor[key].length>0)
   ||!(/^[0-9a-f]{64}$/).test(descriptor.sourceSHA256))return false;
  const shaped=descriptor.version===4;
  return json===canonicalJSON({version:shaped?4:3,providerId:'patreon-v3',providerVersion:descriptor.providerVersion,
   pf2eVersion:descriptor.pf2eVersion,sourceSHA256:descriptor.sourceSHA256,
   qualification:shaped?'patreon-source-shapes.v1':'patreon-original-seams.v1',
   seams:shaped?PATREON_SOURCE_SHAPES:PATREON_SOURCE_SEAMS,markedCommitOwnership:'private-prepare.v1'});
 }catch{return false}
}
