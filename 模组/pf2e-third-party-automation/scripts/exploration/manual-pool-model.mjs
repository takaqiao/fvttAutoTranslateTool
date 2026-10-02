export const MANUAL_POOL_BATCH_SOURCE_SHA='d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157';

export function manualPoolBatchModel(descriptor){
 if(descriptor?.version!==1||descriptor.providerId!=='pf2e'||descriptor.providerVersion!=='8.5.1'||descriptor.protocol!=='pf2e-third-party-automation:manual-pool-batch:1'||descriptor.baseSourceSHA256!==MANUAL_POOL_BATCH_SOURCE_SHA)return false;
 return descriptor.model==='numeric-empty-reception.v1'||descriptor.model==='numeric-static-reception.v1'&&descriptor.staticReceiverModelVersion===1&&descriptor.receiverPredicateModelVersion===1;
}
