window.__log = window.__log || [];
window.__log.push("mod:before-await");
window.addEventListener("DOMContentLoaded", () => window.__log.push("event:DOMContentLoaded"));
const r = await fetch("./data.json");
const j = await r.json();
window.__log.push("mod:after-await value=" + j.v);
