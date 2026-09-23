/** Keep a validated recipient on the damage card across GM/client handoffs. */
export function withDamageMessageTarget(data,targetUuid){
 if(typeof targetUuid!=='string'||!/^Scene\.[^.]+\.Token\.[^.]+$/.test(targetUuid))throw Error('Damage card requires an explicit scene Token UUID.');
 const flags=data.flags??{},toolbelt=flags['pf2e-toolbelt']??{};
 return {...data,flags:{...flags,'pf2e-toolbelt':{...toolbelt,targetHelper:{...toolbelt.targetHelper,targets:[targetUuid]}}}};
}
