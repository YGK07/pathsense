/* PathSense web app: route planner, map layers, problem reports, path conditions editor, walks & data, upload. No build step. */
"use strict";
const $ = (s) => document.querySelector(s);
const COL = ["#2e9e4f", "#f0a020", "#d63a3a"], UNCHECKED = "#8a94a3";
const WALKER = { c01: "#6a3d9a", c02: "#00897b", c03: "#b5651d" };
const CAMPUS = [[12.9665, 79.1505], [12.9770, 79.1695]];
const S = { M: null, R: null, places: [], status: null, profile: "manual_wheelchair", start: null, end: null,
            pick: null, last: null, highlight: null, edit: false };
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* ---------------- helpers ---------------- */
async function api(url, opt) {
  const r = await fetch(url, opt);
  let j = null;
  try { j = await r.json(); } catch (e) { /* non-JSON */ }
  if (!r.ok) throw new Error((j && j.error) || `request failed (${r.status})`);
  return j;
}
let toastTimer;
function toast(msg, ms = 3500) {
  const t = $("#toast"); t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.add("hidden"), ms);
}
const fmtM = (m) => (m >= 1000 ? (m / 1000).toFixed(2) + " km" : Math.round(m) + " m");
const plural = (n, w, ws) => `${n} ${n === 1 ? w : ws || w + "s"}`;
const inCampus = (lat, lon) => lat > CAMPUS[0][0] && lat < CAMPUS[1][0] && lon > CAMPUS[0][1] && lon < CAMPUS[1][1];

/* ---------------- map ---------------- */
const map = L.map("map", { preferCanvas: true, zoomControl: true, renderer: L.canvas({ tolerance: 8 }) });
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "&copy; OpenStreetMap contributors" }).addTo(map);
map.fitBounds(CAMPUS);
// separate panes so redrawing the path layer (on zoom, after an edit) never paints over walks or the route
const pane = (name, z) => { map.createPane(name).style.zIndex = z; return L.canvas({ pane: name, tolerance: 6 }); };
const R_WALKS = pane("walksPane", 410), R_ROUTE = pane("routePane", 420), R_DOTS = pane("dotsPane", 430);
const layers = { access: L.layerGroup().addTo(map), walks: L.layerGroup().addTo(map), kerbs: L.layerGroup().addTo(map),
                 barriers: L.layerGroup().addTo(map), route: L.layerGroup().addTo(map), pins: L.layerGroup().addTo(map) };

function edgeTip(e) {
  const c = e.cond, a = e.est, unchecked = c.surface === "unknown" && a.cls === 0;
  const status = unchecked ? "Not checked yet" : ["Accessible", "Take care", "Not accessible"][a.cls];
  const bits = [`<b>${esc(e.name || "Path")}</b> · ${status}`, ...c.lines.map(esc)];
  (e.reported || []).forEach((r) => bits.push(`reported: ${esc(r)}`));
  if (c.note) bits.push(`team note: ${esc(c.note)}`);
  if (c.photos.length) bits.push(`${plural(c.photos.length, "photo")}`);
  bits.push(e.walks ? `walked by ${plural(e.walks, "contributor")}` : "no recorded walk here yet");
  if (S.edit) bits.push("<i>Click to edit this path</i>");
  return bits.join("<br>");
}
const edgeLayers = new Map();
function edgeStyle(e) {
  const unchecked = e.cond.surface === "unknown" && e.est.cls === 0, hi = S.editing === e.idx;
  return { color: hi ? "#1D5FD1" : unchecked ? UNCHECKED : COL[e.est.cls], weight: hi ? 8 : e.walks ? 5 : 4,
           opacity: unchecked ? 0.65 : 0.9, dashArray: unchecked ? "5 6" : null };
}
function drawAccess() {
  layers.access.clearLayers(); edgeLayers.clear();
  if (!$("#ly-access").checked || !S.M) return;
  const N = new Map(S.M.nodes.map((n) => [n.id, n]));
  for (const e of S.M.edges) {
    const a = N.get(e.u), b = N.get(e.v);
    const pl = L.polyline(e.geom || [[a.lat, a.lon], [b.lat, b.lon]], edgeStyle(e)).bindTooltip(() => edgeTip(e), { sticky: true });
    pl.on("click", (ev) => { if (!S.edit) return; L.DomEvent.stop(ev); openEditor(e); });
    pl.addTo(layers.access); edgeLayers.set(e.idx, pl);
    if (e.cond.photos.length && map.getZoom() >= 17) {
      const mid = (e.geom || [[a.lat, a.lon]])[Math.floor((e.geom || [0]).length / 2)];
      L.marker(mid, { icon: L.divIcon({ className: "", iconSize: [20, 20], html: '<div class="photo-ic">&#128247;</div>' }) })
        .on("click", (ev) => { L.DomEvent.stop(ev); openEditor(e); }).addTo(layers.access);
    }
  }
}
map.on("zoomend", () => { if (S.M) drawAccess(); });
function drawWalks() {
  layers.walks.clearLayers();
  if (!$("#ly-walks").checked || !S.R) return;
  for (const w of S.R.walks) {
    const hi = S.highlight === w.id;
    L.polyline(w.track, { renderer: R_WALKS, color: WALKER[w.contributor] || "#555", weight: hi ? 7 : 3.5, opacity: hi ? 1 : 0.75, dashArray: hi ? null : "2 7" })
      .bindTooltip(`<b>${esc(w.title)}</b><br>${esc(w.contributor)} · ${esc(w.device)} · ${fmtM(w.length_m)}`, { sticky: true }).addTo(layers.walks);
  }
}
function drawKerbs() {
  layers.kerbs.clearLayers();
  if (!$("#ly-kerbs").checked || !S.M) return;
  for (const k of S.M.kerb_spots || [])
    L.circleMarker([k.lat, k.lon], { renderer: R_DOTS, radius: 5, color: "#7a1010", weight: 1.5, fillColor: "#d63a3a", fillOpacity: 0.9 })
      .bindTooltip(`Kerb marked ${plural(k.presses, "time")} on ${plural(k.walks.length, "walk")}`).addTo(layers.kerbs);
  for (const k of S.M.maybe_kerbs || [])
    L.circleMarker([k.lat, k.lon], { renderer: R_DOTS, radius: 4, color: "#a35a00", weight: 1.5, fillColor: "#fff", fillOpacity: 0.9 })
      .bindTooltip(`Possible kerb: a bump the phone felt on ${k.walks} walks, but nobody marked a kerb here`).addTo(layers.kerbs);
}
const BICON = { stairs: "&#9940;", steps: "&#9940;", blocked: "&#9940;", closed: "&#9940;", kerb: "&#9888;", broken: "&#9888;", rough: "&#9888;", steep: "&#9888;" };
const BLABEL = { stairs: "Stairs", steps: "Stairs", kerb: "Kerb", broken: "Broken or rough path", rough: "Broken or rough path",
                 steep: "Steep slope", blocked: "Path blocked or closed", closed: "Path blocked or closed" };
const SOURCE = { team: "reported by the team", walk: "marked on a recorded walk", site: "reported on this site" };
function drawBarriers() {
  layers.barriers.clearLayers();
  if (!$("#ly-barriers").checked || !S.M) return;
  for (const b of S.M.barriers || []) {
    const t = (b.type || "").toLowerCase();
    const m = L.marker([b.lat, b.lon], { icon: L.divIcon({ className: "", iconSize: [22, 22], html: `<div class="barrier-ic">${BICON[t] || "&#9888;"}</div>` }) });
    const del = b.source === "site" ? `<br><button class="linkbtn" data-del="${esc(b.id)}">Remove this report</button>` : "";
    m.bindPopup(`<b>${BLABEL[t] || esc(b.type)}</b>${b.note ? "<br>" + esc(b.note) : ""}<br><span class="note">${SOURCE[b.source] || ""}${b.created ? " · " + esc(b.created) : ""}</span>${del}`);
    m.addTo(layers.barriers);
  }
}
document.addEventListener("click", async (ev) => {
  const id = ev.target?.dataset?.del;
  if (!id) return;
  try { await api(`/api/reports/${encodeURIComponent(id)}`, { method: "DELETE" }); toast("Report removed"); map.closePopup(); await refreshData(true); }
  catch (e) { toast(e.message); }
});
function drawAll() { drawAccess(); drawWalks(); drawKerbs(); drawBarriers(); }
["access", "walks", "kerbs", "barriers"].forEach((k) => $("#ly-" + k).addEventListener("change", drawAll));

/* ---------------- start / destination ---------------- */
function pinIcon(kind) { return L.divIcon({ className: "", iconSize: [22, 22], iconAnchor: [11, 22], html: `<div class="pin ${kind}"></div>` }); }
function setPoint(which, lat, lon, label) {
  S[which] = { lat, lon, label };
  $("#in-" + which).value = label || `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
  drawPins(); $("#go").disabled = !(S.start && S.end);
  if (S.start && S.end) findRoute();
}
function drawPins() {
  layers.pins.clearLayers();
  for (const which of ["start", "end"]) {
    const p = S[which]; if (!p) continue;
    const m = L.marker([p.lat, p.lon], { icon: pinIcon(which), draggable: true, title: which === "start" ? "Start" : "Destination" });
    m.on("dragend", () => { const ll = m.getLatLng(); setPoint(which, ll.lat, ll.lng, "Point on map"); });
    m.addTo(layers.pins);
  }
}
function setPick(mode) {
  if (mode && S.edit) setEditMode(false);
  S.pick = S.pick === mode ? null : mode;
  $("#pick-start").classList.toggle("on", S.pick === "start");
  $("#pick-end").classList.toggle("on", S.pick === "end");
  $("#report").classList.toggle("primary", S.pick === "report");
  const msg = { start: "Click the map to set the start", end: "Click the map to set the destination", report: "Click the map where the problem is" }[S.pick];
  $("#hint").textContent = msg || ""; $("#hint").classList.toggle("hidden", !msg);
  map.getContainer().style.cursor = S.pick ? "crosshair" : "";
}
map.on("click", (e) => {
  const { lat, lng } = e.latlng;
  if (S.edit) return;
  if (S.pick === "start" || S.pick === "end") { const w = S.pick; setPick(null); setPoint(w, lat, lng, "Point on map"); return; }
  if (S.pick === "report") { setPick(null); openReport(lat, lng); return; }
  if (!S.start) setPoint("start", lat, lng, "Point on map");
  else if (!S.end) setPoint("end", lat, lng, "Point on map");
});
$("#pick-start").onclick = () => setPick("start");
$("#pick-end").onclick = () => setPick("end");
$("#swap").onclick = () => { [S.start, S.end] = [S.end, S.start]; $("#in-start").value = S.start?.label || ""; $("#in-end").value = S.end?.label || ""; drawPins(); if (S.start && S.end) findRoute(); };
$("#locate").onclick = () => {
  if (!navigator.geolocation) return toast("This browser can't share your location");
  navigator.geolocation.getCurrentPosition((p) => {
    const { latitude: lat, longitude: lon } = p.coords;
    if (!inCampus(lat, lon)) return toast("You seem to be outside the campus map, so pick the start on the map instead");
    setPoint("start", lat, lon, "My location"); map.setView([lat, lon], 17);
  }, () => toast("Location isn't available here (phones only share it over https). Pick the start on the map instead."), { enableHighAccuracy: true, timeout: 10000 });
};

/* search boxes */
function setupSearch(which) {
  const input = $("#in-" + which), box = $("#sg-" + which);
  let items = [], sel = -1;
  const close = () => { box.classList.add("hidden"); sel = -1; };
  const render = () => {
    box.innerHTML = items.map((p, i) => `<div role="option" data-i="${i}" class="${i === sel ? "sel" : ""}">${esc(p.name)}<small>${esc(p.kind)}</small></div>`).join("");
    box.classList.toggle("hidden", !items.length);
  };
  const choose = (p) => { close(); setPoint(which, p.lat, p.lon, p.name); if (!(S.start && S.end)) map.setView([p.lat, p.lon], 17); };
  input.addEventListener("input", () => {
    const q = input.value.trim().toLowerCase();
    if (!q) { items = []; return render(); }
    const starts = S.places.filter((p) => p.name.toLowerCase().startsWith(q) || (p.alias || "").toLowerCase() === q);
    const has = S.places.filter((p) => !starts.includes(p) && p.name.toLowerCase().includes(q));
    items = starts.concat(has).filter((p) => p.kind !== "road" || q.length > 3).slice(0, 8); sel = items.length ? 0 : -1; render();
  });
  input.addEventListener("keydown", (e) => {
    if (box.classList.contains("hidden")) return;
    if (e.key === "ArrowDown") { sel = Math.min(sel + 1, items.length - 1); render(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { sel = Math.max(sel - 1, 0); render(); e.preventDefault(); }
    else if (e.key === "Enter" && items[sel]) { choose(items[sel]); e.preventDefault(); }
    else if (e.key === "Escape") close();
  });
  box.addEventListener("mousedown", (e) => { const d = e.target.closest("[data-i]"); if (d) { e.preventDefault(); choose(items[+d.dataset.i]); } });
  input.addEventListener("blur", () => setTimeout(close, 150));
}
setupSearch("start"); setupSearch("end");

/* profiles */
function drawProfiles() {
  const order = ["manual_wheelchair", "powered_wheelchair", "crutches", "low_vision", "shortest"];
  $("#profiles").innerHTML = order.filter((k) => S.M.profiles[k]).map((k) =>
    `<button class="chip ${k === S.profile ? "on" : ""}" role="radio" aria-checked="${k === S.profile}" data-p="${k}">${esc(S.M.profiles[k])}</button>`).join("");
}
$("#profiles").addEventListener("click", (e) => {
  const b = e.target.closest("[data-p]"); if (!b) return;
  S.profile = b.dataset.p; drawProfiles(); if (S.start && S.end) findRoute();
});

/* ---------------- routing ---------------- */
$("#go").onclick = () => findRoute(true);
$("#clear").onclick = () => {
  S.start = S.end = S.last = null; $("#in-start").value = $("#in-end").value = "";
  layers.route.clearLayers(); drawPins(); $("#result").classList.add("hidden"); $("#go").disabled = true;
};
$("#demo").onclick = async () => {
  try {
    const d = await api("/api/demo");
    const N = new Map(S.M.nodes.map((n) => [n.id, n]));
    const a = d.from_label, b = d.to_label;
    S.profile = d.profile; drawProfiles();
    S.start = { lat: N.get(d.src).lat, lon: N.get(d.src).lon, label: a };
    $("#in-start").value = a; setPoint("end", N.get(d.dst).lat, N.get(d.dst).lon, b);
  } catch (e) { toast(e.message); }
};
async function findRoute(fit = true) {
  if (!S.start || !S.end) return;
  const q = `/api/route?profile=${S.profile}&from=${S.start.lat},${S.start.lon}&to=${S.end.lat},${S.end.lon}`;
  $("#go").disabled = true; $("#go").textContent = "Finding route…";
  try {
    const r = await api(q); S.last = r; drawRoute(r, fit); showResult(r);
  } catch (e) {
    layers.route.clearLayers(); $("#result").classList.remove("hidden");
    $("#result").innerHTML = `<div class="verdict bad">Couldn't find a route<p>${esc(e.message)}</p></div>`;
  } finally { $("#go").disabled = false; $("#go").textContent = "Find accessible route"; }
}
function drawRoute(r, fit) {
  layers.route.clearLayers();
  if (!r.same) L.polyline(r.shortest.coords, { renderer: R_ROUTE, color: "#222", weight: 4, dashArray: "2 8", opacity: 0.85 }).bindTooltip("Shortest route").addTo(layers.route);
  L.polyline(r.pathsense.coords, { renderer: R_ROUTE, color: "#ffffff", weight: 10, opacity: 0.9 }).addTo(layers.route);
  L.polyline(r.pathsense.coords, { renderer: R_ROUTE, color: "#1D5FD1", weight: 6, opacity: 0.95 }).bindTooltip("PathSense route").addTo(layers.route);
  for (const [p, cls] of [[r.start, "start"], [r.end, "end"]])           // where the trip was snapped onto the paths
    L.circleMarker([p.lat, p.lon], { renderer: R_ROUTE, radius: 5, color: cls === "start" ? "#1E7B45" : "#B42318", fillColor: "#fff", fillOpacity: 1, weight: 3 }).addTo(layers.route);
  if (fit) map.fitBounds(L.latLngBounds(r.pathsense.coords.concat(r.shortest.coords)).pad(0.2), { maxZoom: 18 });
}
function showResult(r) {
  const p = r.pathsense, s = r.shortest, name = S.M.profiles[r.profile];
  const avoided = { kerbs: s.kerbs_found - p.kerbs_found, stairs: s.stairs - p.stairs, problems: s.reported - p.reported, blocked: s.blocked - p.blocked };
  let head, cls;
  if (r.profile === "shortest") { cls = "good"; head = "Shortest walking route"; }
  else if (r.no_accessible_route) { cls = "bad"; head = `No step-free route found for ${name.toLowerCase()}`; }
  else if (r.same) { cls = "good"; head = p.stairs || p.blocked ? "This is the best route we know, but it has barriers" : "The shortest route is already the accessible one"; if (p.stairs || p.blocked) cls = "warn"; }
  else {
    const bits = [];
    if (avoided.stairs > 0) bits.push(plural(avoided.stairs, "staircase", "staircases"));
    if (avoided.kerbs > 0) bits.push(plural(avoided.kerbs, "kerb"));
    if (avoided.problems - avoided.stairs > 0) bits.push(plural(avoided.problems - avoided.stairs, "reported problem"));
    if (s.unpaved_m - p.unpaved_m > 20) bits.push(`${fmtM(s.unpaved_m - p.unpaved_m)} of unpaved path`);
    if (!bits.length && avoided.blocked > 0) bits.push(plural(avoided.blocked, "difficult segment"));
    const extra = p.length - s.length;
    cls = p.stairs || p.closed ? "warn" : "good";
    head = bits.length ? `Avoids ${bits.join(", ")}` : "A smoother route";
    head += extra > 5 ? ` for ${fmtM(extra)} more` : "";
  }
  const sub = r.no_accessible_route ? "Every way between these points has stairs or a closed path. Try a nearby start or destination."
            : p.stairs && r.profile !== "shortest" ? "This route still has stairs."
            : p.unpaved_m > 20 && r.profile !== "shortest" ? `${fmtM(p.unpaved_m)} of it is unpaved, there is no paved way around.` : `About ${Math.max(1, Math.round(p.minutes))} min · ${fmtM(p.length)}.`;
  const row = (lbl, o, sw) => `<tr><td><span class="sw ${sw}"></span>${lbl}</td><td>${fmtM(o.length)}</td><td>${o.kerbs_found}</td><td>${o.stairs}</td><td>${o.unpaved_m ? fmtM(o.unpaved_m) : "0"}</td><td>${100 - o.unchecked_pct}%</td></tr>`;
  const seen = new Map();                                    // the same kind of problem listed once, with a count
  for (const i of (r.same ? p : s).issues) { const k = i.text; seen.has(k) ? (seen.get(k).n += i.n || 1) : seen.set(k, { ...i, n: i.n || 1 }); }
  const issues = [...seen.values()];
  const list = issues.length ? `<div class="label">${r.same ? "On this route" : "On the shortest route (avoided)"}</div><ul class="issues">${issues.slice(0, 8).map((i) =>
      `<li data-ll="${i.lat},${i.lon}">${i.kind === "reported" ? "&#9940;" : i.kind === "kerb" ? "&#9888;" : i.kind === "surface" ? "&#9679;" : "&#8599;"} ${esc(i.text)}${i.n > 1 ? ` <b>×${i.n}</b>` : ""}</li>`).join("")}</ul>` : "";
  $("#result").innerHTML = `<div class="verdict ${cls}">${esc(head)}<p>${esc(sub)}</p></div>
    <table class="cmp"><tr><th>Route</th><th>Length</th><th title="kerbs marked on recorded walks or by the team">Kerbs</th><th>Stairs</th><th>Unpaved</th><th title="share of the route whose surface has been checked">Checked</th></tr>
    ${row("PathSense", p, "ps")}${r.same ? "" : row("Shortest", s, "sh")}</table>${list}
    <p class="note">For ${esc(name.toLowerCase())}. "Checked" is how much of the route has a known surface (team, satellite, OpenStreetMap or campus road).</p>`;
  $("#result").classList.remove("hidden");
  if (innerWidth <= 820) $("#result").scrollIntoView({ behavior: "smooth", block: "start" });   // phones: map on top, result below
}
$("#result").addEventListener("click", (e) => {
  const li = e.target.closest("[data-ll]"); if (!li) return;
  const [lat, lon] = li.dataset.ll.split(",").map(Number); map.setView([lat, lon], 18);
});
$("#fit").onclick = () => map.fitBounds(CAMPUS);
$("#fit").textContent = innerWidth <= 820 ? "Whole campus" : "Show whole campus";
if (innerWidth <= 820) { $("#report").innerHTML = "&#9888; Report"; $("#editpaths").innerHTML = "&#9998; Edit"; }

/* ---------------- report a problem ---------------- */
const RTYPES = [["stairs", "Stairs"], ["kerb", "Kerb"], ["broken", "Broken path"], ["steep", "Steep slope"], ["blocked", "Blocked / closed"]];
let reportAt = null, reportType = null;
$("#report").onclick = () => { if (location.hash && location.hash !== "#/") location.hash = "#/"; setPick("report"); };
function openReport(lat, lon) {
  reportAt = { lat, lon }; reportType = null;
  $("#m-types").innerHTML = RTYPES.map(([k, l]) => `<button class="chip" data-t="${k}">${l}</button>`).join("");
  $("#m-where").textContent = `At ${lat.toFixed(5)}, ${lon.toFixed(5)}`;
  $("#m-note").value = ""; $("#m-err").classList.add("hidden"); $("#modal").classList.remove("hidden");
}
$("#m-types").addEventListener("click", (e) => {
  const b = e.target.closest("[data-t]"); if (!b) return;
  reportType = b.dataset.t; [...$("#m-types").children].forEach((c) => c.classList.toggle("on", c === b));
});
$("#m-cancel").onclick = () => $("#modal").classList.add("hidden");
$("#modal").addEventListener("click", (e) => { if (e.target.id === "modal") $("#modal").classList.add("hidden"); });
$("#m-save").onclick = async () => {
  const err = $("#m-err");
  if (!reportType) { err.textContent = "Choose what the problem is."; return err.classList.remove("hidden"); }
  $("#m-save").disabled = true;
  try {
    const r = await api("/api/reports", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: reportType, lat: reportAt.lat, lon: reportAt.lon, note: $("#m-note").value, contributor: $("#m-contrib").value }) });
    $("#modal").classList.add("hidden"); toast(`${r.label} reported. Routes now take it into account.`);
    await refreshData(true);
  } catch (e) { err.textContent = e.message; err.classList.remove("hidden"); }
  finally { $("#m-save").disabled = false; }
};

/* ---------------- path conditions editor ---------------- */
const ED = { surface: [["smooth", "Smooth paving"], ["paved", "Paved"], ["broken", "Broken or patchy"], ["unpaved", "Unpaved (dirt, gravel, grass)"], ["unknown", "Don't know"]],
             kerb: [["none", "No kerbs"], ["kerb", "Has a kerb"], ["ramp", "Kerbs have ramps"], ["unknown", "Don't know"]],
             steps: [["no", "No steps"], ["yes", "Has steps"], ["unknown", "Don't know"]],
             slope: [["flat", "Flat"], ["gentle", "Gentle slope"], ["steep", "Steep"], ["unknown", "Don't know"]] };
let edEdge = null;
function setEditMode(on) {
  S.edit = on; if (on) setPick(null);
  $("#editpaths").classList.toggle("primary", on);
  $("#hint").textContent = on ? "Click any path to record what it is really like" : ""; $("#hint").classList.toggle("hidden", !on);
  if (on && !$("#ly-access").checked) { $("#ly-access").checked = true; drawAccess(); }
}
$("#editpaths").onclick = () => { if (location.hash && location.hash !== "#/") location.hash = "#/"; setEditMode(!S.edit); };
function openEditor(e) {
  edEdge = e; const t = (S.M.team || {})[String(e.idx)] || {}, c = e.cond;
  S.editing = e.idx; edgeLayers.forEach((pl, i) => pl.setStyle(edgeStyle(S.M.edges.find((x) => x.idx === i) || e)));
  $("#e-title").textContent = e.name || "Path segment " + e.idx;
  $("#e-where").textContent = `${fmtM(e.length)} · ${c.highway || "path"} · segment ${e.idx}${e.walks ? " · walked by " + plural(e.walks, "contributor") : ""}`;
  $("#e-now").innerHTML = c.lines.map((l) => `<li>${esc(l)}</li>`).join("") + (e.reported || []).map((r) => `<li>reported: ${esc(r)}</li>`).join("") +
    (c.satellite_note && c.surface_src === "satellite" ? `<li>satellite: ${esc(c.satellite_note)}</li>` : "") +
    (c.team ? `<li>team edit${c.team_by ? " by " + esc(c.team_by) : ""}${c.team_at ? ", " + esc(c.team_at) : ""}${c.note ? ': "' + esc(c.note) + '"' : ""}</li>` : "");
  for (const [k, opts] of Object.entries(ED)) {
    const cur = t[k] || "unknown";
    $("#e-" + k).innerHTML = opts.map(([v, l]) => `<button type="button" class="chip ${v === cur ? "on" : ""}" data-v="${v}">${l}</button>`).join("");
  }
  $("#e-narrow").checked = !!t.narrow; $("#e-note").value = t.note || ""; $("#e-photo").value = "";
  $("#e-photos").innerHTML = c.photos.map((p) => `<a href="/photos/${esc(p)}" target="_blank" rel="noopener"><img src="/photos/${esc(p)}" alt="Photo of this path"></a>`).join("");
  $("#e-clear").classList.toggle("hidden", !c.team); $("#e-err").classList.add("hidden");
  $("#edmodal").classList.remove("hidden");
}
function closeEditor() { $("#edmodal").classList.add("hidden"); S.editing = null; edgeLayers.forEach((pl, i) => pl.setStyle(edgeStyle(S.M.edges.find((x) => x.idx === i)))); }
for (const k of Object.keys(ED))
  $("#e-" + k).addEventListener("click", (ev) => { const b = ev.target.closest("[data-v]"); if (!b) return;
    [...$("#e-" + k).children].forEach((x) => x.classList.toggle("on", x === b)); });
$("#e-cancel").onclick = closeEditor;
$("#edmodal").addEventListener("click", (ev) => { if (ev.target.id === "edmodal") closeEditor(); });
$("#e-save").onclick = async () => {
  const fd = new FormData(), err = $("#e-err");
  for (const k of Object.keys(ED)) fd.append(k, $("#e-" + k).querySelector(".on")?.dataset.v || "unknown");
  if ($("#e-narrow").checked) fd.append("narrow", "1");
  fd.append("note", $("#e-note").value); fd.append("contributor", $("#e-contrib").value.trim().toLowerCase());
  if ($("#e-photo").files[0]) fd.append("photo", $("#e-photo").files[0]);
  $("#e-save").disabled = true; $("#e-save").textContent = "Saving…";
  try {
    await api(`/api/conditions/${edEdge.idx}`, { method: "POST", body: fd });
    closeEditor(); toast("Path updated. Routes now use what you recorded."); await refreshData(true);
  } catch (e) { err.textContent = e.message; err.classList.remove("hidden"); }
  finally { $("#e-save").disabled = false; $("#e-save").textContent = "Save"; }
};
$("#e-clear").onclick = async () => {
  try { await api(`/api/conditions/${edEdge.idx}`, { method: "DELETE" }); closeEditor(); toast("Team edit removed"); await refreshData(true); }
  catch (e) { $("#e-err").textContent = e.message; $("#e-err").classList.remove("hidden"); }
};

/* ---------------- walks & data page ---------------- */
function drawWalksPage() {
  const sm = S.R?.summary || {}, st = S.status || {};
  const stat = (n, l) => `<div class="stat"><b>${n}</b><span>${l}</span></div>`;
  $("#stats").innerHTML = stat(sm.n_walks || 0, "walks recorded") + stat((sm.total_km || 0) + " km", "walked") +
    stat(sm.n_contributors || 0, "contributors") + stat(sm.n_devices || 0, "phone models") +
    stat(sm.edges_walked || 0, "path segments walked") + stat(st.kerbs_marked || 0, "kerb spots marked on walks") +
    stat((sm.labelled_up || 0) + (sm.labelled_down || 0), "kerb crossings marked") + stat(st.barriers || 0, "stairs and problems on the map");
  const cv = S.M?.coverage || {}, n = cv.segments || 1, pct = (k) => Math.round(100 * (cv[k] || 0) / n);
  $("#coverage").innerHTML = [["team", "Checked by the team on this site"], ["satellite", "Satellite imagery check"],
    ["osm", "OpenStreetMap surface tag"], ["road", "Campus road (taken as paved)"], ["track", "OpenStreetMap track (taken as unpaved)"],
    ["none", "Not checked yet"]].map(([k, l]) => `<tr><td>${l}</td><td>${cv[k] || 0}</td><td><div class="bar"><span style="width:${pct(k)}%"></span></div></td><td>${pct(k)}%</td></tr>`).join("");
  const rows = (S.R?.walks || []).slice().sort((a, b) => (b.date || "").localeCompare(a.date || ""));
  $("#walktable").innerHTML = `<tr><th>Walk</th><th>By</th><th>Phone</th><th>Date</th><th>Length</th><th>Time</th><th>Kerbs marked</th><th>Stairs</th><th>Synced</th><th></th></tr>` +
    rows.map((w) => `<tr><td><b>${esc(w.title)}</b></td><td><span class="sw" style="background:${WALKER[w.contributor] || "#555"}"></span>${esc(w.contributor)}</td>
      <td>${esc(w.device)}</td><td>${esc(w.date)}</td><td>${fmtM(w.length_m)}</td><td>${Math.round(w.duration_s / 60)} min</td>
      <td>${w.labelled.length}</td><td>${w.stairs.length}</td><td>${w.sync_ok ? '<span class="tag ok">yes</span>' : '<span class="tag no">no</span>'}</td>
      <td><button class="btn ghost" data-show="${esc(w.id)}">Show on map</button></td></tr>`).join("");
  const B = S.M?.barriers || [];
  $("#reporttable").innerHTML = `<tr><th>Problem</th><th>Details</th><th>Source</th><th>Where</th><th></th></tr>` + (B.length ? B.map((b) =>
    `<tr><td><b>${BLABEL[(b.type || "").toLowerCase()] || esc(b.type)}</b></td><td>${esc(b.note || "")}</td><td>${SOURCE[b.source] || ""}</td>
     <td><button class="linkbtn" data-goto="${b.lat},${b.lon}">${b.lat.toFixed(5)}, ${b.lon.toFixed(5)}</button></td>
     <td>${b.source === "site" ? `<button class="linkbtn" data-del="${esc(b.id)}">Remove</button>` : ""}</td></tr>`).join("") : `<tr><td colspan="5">Nothing reported yet.</td></tr>`);
}
document.addEventListener("click", (e) => {
  const sh = e.target.closest("[data-show]"), go = e.target.closest("[data-goto]");
  if (sh) { S.highlight = sh.dataset.show; location.hash = "#/"; $("#ly-walks").checked = true; drawWalks();
    const w = S.R.walks.find((x) => x.id === S.highlight); if (w) setTimeout(() => map.fitBounds(L.latLngBounds(w.track).pad(0.15)), 60); }
  if (go) { location.hash = "#/"; const [lat, lon] = go.dataset.goto.split(",").map(Number); setTimeout(() => map.setView([lat, lon], 18), 60); }
});

/* ---------------- upload ---------------- */
let upFile = null;
const drop = $("#drop"), fileIn = $("#file");
const pickFile = (f) => { upFile = f; drop.querySelector("b").textContent = f ? f.name : "Drop the Sensor Logger .zip here"; checkUpload(); };
const checkUpload = () => { $("#upload").disabled = !(upFile && /^c\d{2,3}$/i.test($("#up-contrib").value.trim())); };
drop.onclick = () => fileIn.click();
drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileIn.click(); } };
fileIn.onchange = () => pickFile(fileIn.files[0]);
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) pickFile(f); });
$("#up-contrib").addEventListener("input", checkUpload);
$("#upload").onclick = async () => {
  const res = $("#upres"); const fd = new FormData();
  fd.append("file", upFile); fd.append("contributor", $("#up-contrib").value.trim().toLowerCase()); fd.append("carry", $("#up-carry").value);
  $("#upload").disabled = true; $("#upload").textContent = "Processing the walk… (up to a minute)"; res.innerHTML = "";
  try {
    const r = await api("/api/upload", { method: "POST", body: fd });
    const w = r.walk;
    res.innerHTML = `<div class="verdict good">Walk added: ${esc(w?.title || upFile.name)}<p>${w ? `${fmtM(w.length_m)} in ${Math.round(w.duration_s / 60)} min · ${plural(w.kerbs_marked, "kerb")} marked · ${plural(w.stairs, "staircase")} marked · ${plural(w.jolts, "jolt")} detected · ${w.synced ? "button presses synced" : "button presses could not be synced (the 3 start or end presses were not found)"}` : ""}. The map now has ${r.summary.n_walks} walks.</p></div>`;
    pickFile(null); fileIn.value = ""; await refreshData(true);
  } catch (e) { res.innerHTML = `<div class="verdict bad">Upload failed<p>${esc(e.message)}</p></div>`; }
  finally { $("#upload").textContent = "Upload and update the map"; checkUpload(); }
};

/* ---------------- views & data ---------------- */
function route_() {
  const v = (location.hash.replace(/^#\/?/, "") || "map").split("?")[0];
  const view = ["walks", "contribute", "about"].includes(v) ? v : "map";
  document.querySelectorAll(".view").forEach((s) => s.classList.toggle("hidden", s.id !== "view-" + view));
  document.querySelectorAll("nav.tabs a").forEach((a) => a.classList.toggle("active", a.dataset.view === view));
  if (view === "map") setTimeout(() => map.invalidateSize(), 30);
  if (view === "walks") drawWalksPage();
}
window.addEventListener("hashchange", route_);

async function refreshData(reroute) {
  const [m, r, st, team] = await Promise.all([api("/api/map"), api("/api/real"), api("/api/status"), api("/api/conditions")]);
  S.M = m; S.R = r; S.status = st; S.M.team = team;
  $("#statuspill").textContent = `${st.walks} walks · ${st.km} km · ${st.kerbs_marked} kerbs marked`;
  drawAll(); drawWalksPage();
  if (reroute && S.start && S.end) findRoute(false);
}
(async function init() {
  try {
    const [places] = await Promise.all([api("/api/places"), refreshData(false)]);
    S.places = places; drawProfiles(); route_();
    setTimeout(() => { map.invalidateSize(); map.fitBounds(CAMPUS); }, 80);
    if (new URLSearchParams(location.search).has("demo")) $("#demo").click();
  } catch (e) {
    $("#statuspill").textContent = "Server not reachable";
    toast("Couldn't load the map: " + e.message, 8000);
  }
})();
