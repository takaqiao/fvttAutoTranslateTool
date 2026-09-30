// PF2e documents saved before typed compendium UUIDs retain the same item ID.
export const canonicalItemSource=value=>typeof value==='string'?value.replace(/^(Compendium\.[^.]+\.[^.]+)\.([A-Za-z0-9]{16})$/,'$1.Item.$2'):value;
