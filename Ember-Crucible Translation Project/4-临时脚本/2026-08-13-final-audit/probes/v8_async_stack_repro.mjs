class Actor {
  static async updateDocuments(u) {
    const s = String(new Error().stack);
    console.log("--- stack seen inside updateDocuments ---");
    console.log(s);
    console.log("includes Adventure.importContent:", s.includes("Adventure.importContent"));
    console.log("includes EmberAdventureImporter._processSubmitData:", s.includes("EmberAdventureImporter._processSubmitData"));
    return [];
  }
}
class AdventureImporter {
  async _processSubmitData() { return this.doImport(); }
  async doImport() { return this.adv.import(); }
}
class EmberAdventureImporter extends AdventureImporter {}
class Adventure {
  async import() {
    await new Promise(r => setTimeout(r, 1));   // real macrotask boundary
    return this.importContent({});
  }
  async importContent(data) {
    await new Promise(r => setTimeout(r, 1));   // resumption inside importContent
    const cls = Actor;
    return await cls.updateDocuments([{}]);
  }
}
const imp = new EmberAdventureImporter();
imp.adv = new Adventure();
await imp._processSubmitData();
