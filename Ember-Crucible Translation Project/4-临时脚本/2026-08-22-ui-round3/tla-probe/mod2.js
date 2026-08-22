window.__log2 = [];
window.__log2.push("mod2:start");
window.addEventListener("DOMContentLoaded", () => window.__log2.push("event:DOMContentLoaded"));
try {
  const xhr = new XMLHttpRequest();
  xhr.open("GET", "./data.json", false);   // sync
  xhr.send(null);
  window.__log2.push("mod2:sync-xhr status=" + xhr.status + " value=" + JSON.parse(xhr.responseText).v);
} catch (err) {
  window.__log2.push("mod2:sync-xhr THREW " + err);
}
window.__log2.push("mod2:end");
