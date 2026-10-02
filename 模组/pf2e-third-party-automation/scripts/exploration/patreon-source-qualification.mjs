import {canonicalJSON} from './revision-codec.mjs';

export const PATREON_SOURCE_SEAMS=Object.freeze({
 time:'8c32f064ac33ab4a0ce264c8d74c8acbf8621627f1a8ef6f111e9ecb606cfa92',
 manualImmunity:'bdc211e3b7e437e91f8d9ab2c8f54c6b189abd8789278989de3a5a5b84e566a0',
 createItem:'14162e56764de7549e40aa9dad55249c14fe072acf181045f288bd02bd9bd7df'
});

export function isPatreonSourceQualified(value){
 try{
  const json=canonicalJSON(value),descriptor=JSON.parse(json);
  if(Reflect.ownKeys(value).length!==8||Reflect.ownKeys(value.seams).length!==3
   ||!['providerVersion','pf2eVersion'].every(key=>typeof descriptor[key]==='string'&&descriptor[key].length>0)
   ||!(/^[0-9a-f]{64}$/).test(descriptor.sourceSHA256))return false;
  return json===canonicalJSON({version:3,providerId:'patreon-v3',providerVersion:descriptor.providerVersion,
   pf2eVersion:descriptor.pf2eVersion,sourceSHA256:descriptor.sourceSHA256,qualification:'patreon-original-seams.v1',
   seams:PATREON_SOURCE_SEAMS,markedCommitOwnership:'private-prepare.v1'});
 }catch{return false}
}
