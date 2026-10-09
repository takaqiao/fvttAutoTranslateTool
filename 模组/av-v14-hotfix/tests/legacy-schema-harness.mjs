import {pathToFileURL} from 'node:url';

const app=process.env.FVTT_NATIVE_APP??'C:/Program Files/Foundry Virtual Tabletop/resources/app';
const common=pathToFileURL(`${app}/common/`).href;
globalThis.CONST=await import(common+'constants.mjs');
globalThis.foundry={abstract:await import(common+'abstract/_module.mjs'),
  data:await import(common+'data/_module.mjs'),documents:await import(common+'documents/_module.mjs')};
globalThis.CONFIG??={};
const {BaseChatMessage,BaseScene,BaseLevel}=foundry.documents;

// Real schema definitions and field classes; document persistence is supplied by each test.
export function attachLegacySchemas(g) {
  class Level extends BaseLevel {static get implementation(){return this;}}
  Object.defineProperty(Level,'schema',{value:new foundry.data.fields.DataModelSchemaField(Level)});
  class SceneSchema extends BaseScene {
    static defineSchema(){return {...super.defineSchema(),levels:new foundry.data.fields.EmbeddedCollectionField(Level)};}
  }
  class MessageSchema extends BaseChatMessage {}
  g.foundry={data:{fields:foundry.data.fields}};
  g.CONFIG.Level={documentClass:Level};
  for(const [target,native]of [[g.ChatMessage,MessageSchema],[g.Scene,SceneSchema]]) {
    if(!target)continue;
    Object.defineProperties(target,{
      documentName:{value:native.documentName,configurable:true},
      metadata:{value:native.metadata,configurable:true},
      schema:{value:new foundry.data.fields.DataModelSchemaField(native),configurable:true}
    });
  }
  return g;
}
