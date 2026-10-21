"""PathSense web app.

    python app/server.py            # http://127.0.0.1:5000 on this computer
    python app/server.py --lan      # also reachable from phones on the same Wi-Fi (http://<this PC's IP>:5000)

On first start it builds the campus map (scripts/run_pipeline.py) and the real-walks layer (scripts/process_walks.py)
if they are missing. Everything the site needs lives under the project folder:
  outputs/map.json          campus path map (OpenStreetMap paths + estimated surface)
  outputs/real_walks.json   recorded walks, kerb jolts, confirmed kerb spots, stairs (from data/raw/)
  data/osm_places.json      named campus places for the search box
  data/reports.json         problems reported on the site ("Report a problem")
  data/ground_truth/features.csv   barriers reported by the team, with a position
  data/path_conditions.json  path conditions the team set on the site ("Edit paths"), photos in data/photos/
  data/satellite_review.json surfaces checked on satellite imagery
Path conditions come only from real sources (see pathsense/conditions.py); the simulator's values are not used here.
"""
import argparse, copy, csv, io, json, os, re, socket, subprocess, sys, threading, time, uuid, zipfile
import networkx as nx
from flask import Flask, Response, abort, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from pathsense.routing import route, path_stats, mark_reported, REPORT_TYPES
from pathsense import conditions as CND
from pathsense.config import PROFILES, CLASSES
from pathsense.osm import edge_coords, load_graph
from process_walks import process, find_walks, load_contributors, load_options

PROFILE_LABELS = {"manual_wheelchair": "Manual wheelchair", "powered_wheelchair": "Powered wheelchair",
                  "crutches": "Crutches or walking aid", "low_vision": "Low vision", "shortest": "Walking (shortest)"}
# The fixed trip behind the "Demo route" button: the shortest path goes through the TT Subway stairs.
DEMO = dict(start=(12.97305, 79.16405), end=(12.97100, 79.15843), profile="manual_wheelchair",
            from_label="Fitty gym (hostel side)", to_label="Souvenier Subway (academic side)")
REPORT_KINDS = ("stairs", "kerb", "broken", "steep", "blocked")
EDIT_CHOICES = dict(surface=("smooth", "paved", "broken", "unpaved", "unknown"), kerb=("none", "kerb", "ramp", "unknown"),
                    steps=("no", "yes", "unknown"), slope=("flat", "gentle", "steep", "unknown"))
PHOTO_TYPES = {b"\xff\xd8\xff": ".jpg", b"\x89PNG": ".png"}


def build_graph(m):
    G = nx.Graph()
    for n in m["nodes"]:
        G.add_node(n["id"], x=n["x"], y=n["y"], lat=n["lat"], lon=n["lon"])
    for e in m["edges"]:
        t = e["true"]
        G.add_edge(e["u"], e["v"], idx=e["idx"], length=e["length"], est=dict(e["est"]), observed=e["observed"],
                   grad=t["grad"], rough=t["rough"], kerbs=t["kerbs"], cls=t["cls"], name=e.get("name"),
                   geom=e.get("geom"), geom_from=e.get("geom_from"),
                   **{k: v for k, v in e.items() if k.startswith("osm_") or k == "steps"})
    return G


class World:
    """Everything the site serves, rebuilt as a whole after an upload or a report (swapped in one assignment)."""

    def __init__(self, paths):
        self.p = paths
        self.lock = threading.Lock()
        self.osm_tags = {}
        if os.path.exists(paths["osm"]):                     # road type etc. straight from OpenStreetMap, by edge idx
            for k, (*_, d) in enumerate(load_graph(paths["osm"]).edges(data=True)):
                self.osm_tags[k] = {a: v for a, v in d.items() if a.startswith("osm_") or a == "steps"}
        self.rebuild()

    def _json(self, key, default):
        path = self.p[key]
        return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else default

    def site_reports(self):
        return self._json("reports", [])

    def team_reports(self):
        out = []
        if os.path.exists(self.p["features"]):
            for r in csv.DictReader(open(self.p["features"], encoding="utf-8")):
                try:
                    out.append(dict(id=r["feature_id"], type=r["type"], lat=float(r["lat"]), lon=float(r["lon"]),
                                    note=r.get("landmark", ""), source="team"))
                except (TypeError, ValueError, KeyError):
                    continue
        return out

    def walk_stairs(self, real):
        return [dict(id=f"walk-{w['id']}-{k}", type="stairs", lat=s["lat"], lon=s["lon"],
                     note=f"stairs {s['direction']} marked on '{w.get('title', w['id'])}'", source="walk")
                for w in real.get("walks", []) for k, s in enumerate(w.get("stairs", []))]

    def rebuild(self):
        m = self._json("map", None)
        if m is None:
            raise FileNotFoundError(self.p["map"])
        real = self._json("real", {"walks": [], "clusters": [], "summary": {}})
        G = build_graph(m)
        for *_, d in G.edges(data=True):
            d.update(self.osm_tags.get(d["idx"], {}))
        team, sat = self._json("conditions", {}), self._json("satellite", {})
        cond, spots, maybes = CND.build(G, real, team, sat)
        for *_, d in G.edges(data=True):
            c = cond[d["idx"]]
            d["cond"] = c; d["est"] = CND.attrs(c); d["real_kerbs"] = c["kerbs"]
        barriers = self.team_reports() + self.walk_stairs(real) + [dict(r, source="site") for r in self.site_reports()]
        mark_reported(G, barriers)
        walked = {int(k): len(v) for k, v in real.get("edge_walks", {}).items()}
        for *_, d in G.edges(data=True):
            d["walks"] = walked.get(d["idx"], 0)
        # the map the browser draws: the same edges, with what routing now believes
        by_idx = {d["idx"]: d for *_, d in G.edges(data=True)}
        view = copy.deepcopy(m)
        for e in view["edges"]:
            d = by_idx[e["idx"]]
            e["est"] = d["est"]; e["real_kerbs"] = d.get("real_kerbs", 0)
            e["reported"] = d.get("reported", []); e["walks"] = d["walks"]
            e["cond"] = dict(d["cond"], lines=CND.describe(d["cond"]))
            for k in ("true", "src", "observed", "n_obs"):    # simulator values are not shown
                e.pop(k, None)
        view["profiles"] = {k: PROFILE_LABELS[k] for k in PROFILES}
        view["barriers"] = barriers
        view["kerb_spots"] = spots; view["maybe_kerbs"] = maybes
        view["coverage"] = {src or "none": sum(c["surface_src"] == src for c in cond.values())
                            for src in ("team", "satellite", "osm", "road", "track", None)}
        view["coverage"]["segments"] = len(cond)
        self.G, self.real, self.view, self.barriers, self.team = G, real, view, barriers, team
        self.places = self._json("places", [])
        self.built = time.strftime("%Y-%m-%d %H:%M:%S")

    def nearest_node(self, lat, lon):
        G = self.G
        return min(G.nodes, key=lambda n: (G.nodes[n]["lat"] - lat) ** 2 + (G.nodes[n]["lon"] - lon) ** 2)


def describe(G, path):
    """Route stats for the page, from what PathSense believes (estimated map + real kerbs + reports)."""
    s = path_stats(G, path, "shortest", "est")
    issues = []
    for u, v in zip(path[:-1], path[1:]):
        d = G[u][v]; a = d["est"]
        mid = (d["geom"][len(d["geom"]) // 2] if d.get("geom") else
               [(G.nodes[u]["lat"] + G.nodes[v]["lat"]) / 2, (G.nodes[u]["lon"] + G.nodes[v]["lon"]) / 2])
        c = d.get("cond", {})
        for r in d.get("reported", []):
            issues.append(dict(kind="reported", text=r, lat=mid[0], lon=mid[1]))
        if c.get("steps"):
            issues.append(dict(kind="reported", text="steps", lat=mid[0], lon=mid[1]))
        if d.get("real_kerbs"):
            issues.append(dict(kind="kerb", text="kerb marked on recorded walks" if c.get("kerb_walks") else "kerb",
                               lat=mid[0], lon=mid[1], n=d["real_kerbs"]))
        if c.get("surface") == "unpaved":
            issues.append(dict(kind="surface", text=f"unpaved path ({round(d['length'])} m)", lat=mid[0], lon=mid[1]))
        elif c.get("surface") == "broken":
            issues.append(dict(kind="surface", text=f"broken paving ({round(d['length'])} m)", lat=mid[0], lon=mid[1]))
        if c.get("maybe"):
            issues.append(dict(kind="kerb", text="possible kerb (bump found by the phone)", lat=mid[0], lon=mid[1], n=c["maybe"]))
        if a.get("grad", 0) >= 7:
            issues.append(dict(kind="steep", text=f"steep slope (about {a['grad']:.0f}°)", lat=mid[0], lon=mid[1]))
    return dict(length=round(s["length"]), kerbs_found=s["real_kerbs"], stairs=s["stairs"], closed=s["closed"],
                reported=s["reported"], blocked=s["blocked"], caution=s["caution"],
                walked_pct=round(100 * s["walked_frac"]), mean_rough=round(s["mean_rough"], 2),
                unpaved_m=round(s["unpaved_m"]), unchecked_pct=round(100 * s["unknown_m"] / max(s["length"], 1)),
                maybe_kerbs=s["maybe_kerbs"],
                minutes=round(s["length"] / 70.0, 1), issues=issues)


def create_app(root=ROOT, auto_build=True):
    paths = dict(map=os.path.join(root, "outputs", "map.json"), real=os.path.join(root, "outputs", "real_walks.json"),
                 places=os.path.join(root, "data", "osm_places.json"), reports=os.path.join(root, "data", "reports.json"),
                 features=os.path.join(root, "data", "ground_truth", "features.csv"),
                 raw=os.path.join(root, "data", "raw"), osm=os.path.join(root, "data", "osm_vit.json"),
                 conditions=os.path.join(root, "data", "path_conditions.json"),
                 satellite=os.path.join(root, "data", "satellite_review.json"), photos=os.path.join(root, "data", "photos"))
    if auto_build and not os.path.exists(paths["map"]):
        print("Building the campus map (first start, about a minute)...", flush=True)
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_pipeline.py"), "--osm", paths["osm"],
                        "--out", os.path.dirname(paths["map"])], check=True, stdout=subprocess.DEVNULL)
    if auto_build and not os.path.exists(paths["real"]) and os.path.isdir(paths["raw"]):
        print("Processing recorded walks...", flush=True)
        os.makedirs(os.path.dirname(paths["real"]), exist_ok=True)
        process(find_walks(paths["raw"]), paths["osm"], paths["real"],
                load_contributors(os.path.join(paths["raw"], "contributors.csv")),
                load_options(os.path.join(paths["raw"], "walk_options.json")))

    app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024          # a 10-minute walk zip is ~6 MB
    world = World(paths)
    app.config["WORLD"] = world

    def bad(msg, code=400):
        return jsonify(error=msg), code

    def page():
        """index.html with ?v=<file time> on our CSS/JS, so browsers never run an old copy after an update."""
        html = open(os.path.join(app.static_folder, "index.html"), encoding="utf-8").read()
        for f in ("app.css", "app.js"):
            v = int(os.path.getmtime(os.path.join(app.static_folder, f)))
            html = html.replace(f"/static/{f}", f"/static/{f}?v={v}")
        return Response(html, mimetype="text/html", headers={"Cache-Control": "no-cache"})

    @app.get("/")
    def index():
        return page()

    @app.get("/api/status")
    def api_status():
        w = app.config["WORLD"]; S = w.real.get("summary", {})
        return jsonify(ok=True, built=w.built, segments=len(w.view["edges"]), places=len(w.places),
                       walks=S.get("n_walks", 0), km=S.get("total_km", 0), contributors=S.get("n_contributors", 0),
                       kerbs_marked=len(w.view["kerb_spots"]), barriers=len(w.barriers), coverage=w.view["coverage"])

    @app.get("/api/map")
    def api_map():
        return jsonify(app.config["WORLD"].view)

    @app.get("/api/real")
    def api_real():
        return jsonify(app.config["WORLD"].real)

    @app.get("/api/places")
    def api_places():
        return jsonify(app.config["WORLD"].places)

    @app.get("/api/demo")
    def api_demo():
        w = app.config["WORLD"]
        return jsonify(src=w.nearest_node(*DEMO["start"]), dst=w.nearest_node(*DEMO["end"]),
                       profile=DEMO["profile"], from_label=DEMO["from_label"], to_label=DEMO["to_label"])

    def node_arg(name):
        """?src=<node id> or ?from=lat,lon (snapped to the nearest junction)."""
        w = app.config["WORLD"]
        v = request.args.get(name)
        if v is None:
            return None
        if "," in v:
            lat, lon = (float(x) for x in v.split(",", 1))
            return w.nearest_node(lat, lon)
        n = int(v)
        if n not in w.G:
            raise ValueError(f"unknown junction {n}")
        return n

    @app.get("/api/route")
    def api_route():
        w = app.config["WORLD"]
        profile = request.args.get("profile", "manual_wheelchair")
        if profile not in PROFILES:
            return bad(f"unknown profile '{profile}'")
        try:
            src = node_arg("src") if "src" in request.args else node_arg("from")
            dst = node_arg("dst") if "dst" in request.args else node_arg("to")
        except ValueError as e:
            return bad(str(e))
        if src is None or dst is None:
            return bad("give a start and a destination")
        if src == dst:
            return bad("start and destination are the same place")
        G = w.G
        try:
            ps = route(G, src, dst, profile, "est")
            sh = route(G, src, dst, "shortest", "est")
        except nx.NetworkXNoPath:
            return bad("no route between these two points on the campus paths", 404)
        p, s = describe(G, ps), describe(G, sh)
        blocked = profile != "shortest" and (p["stairs"] and PROFILES[profile]["w_s"] >= 1e5 or p["closed"])
        node = lambda n: dict(id=n, lat=G.nodes[n]["lat"], lon=G.nodes[n]["lon"])
        return jsonify(profile=profile, start=node(src), end=node(dst), same=(ps == sh), no_accessible_route=bool(blocked),
                       pathsense=dict(coords=edge_coords(G, ps), **p), shortest=dict(coords=edge_coords(G, sh), **s))

    @app.get("/api/reports")
    def api_reports():
        return jsonify(app.config["WORLD"].barriers)

    @app.post("/api/reports")
    def api_report_add():
        w = app.config["WORLD"]
        d = request.get_json(silent=True) or {}
        kind = str(d.get("type", "")).lower()
        if kind not in REPORT_KINDS:
            return bad("choose what the problem is: " + ", ".join(REPORT_KINDS))
        try:
            lat, lon = float(d["lat"]), float(d["lon"])
        except (KeyError, TypeError, ValueError):
            return bad("click the map where the problem is")
        note = re.sub(r"\s+", " ", str(d.get("note", ""))).strip()[:200]
        contributor = str(d.get("contributor", "")).strip().lower()
        if contributor and not re.fullmatch(r"c\d{2,3}", contributor):
            return bad("contributor id must look like c01 (no names, please)")
        G = w.G
        from pathsense.io import edge_index, match_points
        e, dist = match_points(edge_index(G), [lat], [lon], max_dist=25.0, smooth=1)
        if e[0] < 0:
            return bad("that spot is more than 25 m from any mapped path; click closer to the path")
        rep = dict(id=uuid.uuid4().hex[:10], type=kind, lat=round(lat, 6), lon=round(lon, 6), note=note,
                   contributor=contributor, created=time.strftime("%Y-%m-%d %H:%M"))
        with w.lock:
            reps = w.site_reports(); reps.append(rep)
            os.makedirs(os.path.dirname(w.p["reports"]), exist_ok=True)
            json.dump(reps, open(w.p["reports"], "w", encoding="utf-8"), indent=1)
            w.rebuild()
        return jsonify(ok=True, report=rep, label=REPORT_TYPES[kind])

    @app.delete("/api/reports/<rid>")
    def api_report_delete(rid):
        w = app.config["WORLD"]
        with w.lock:
            reps = w.site_reports()
            keep = [r for r in reps if r["id"] != rid]
            if len(keep) == len(reps):
                return bad("only problems reported on this site can be removed here", 404)
            json.dump(keep, open(w.p["reports"], "w", encoding="utf-8"), indent=1)
            w.rebuild()
        return jsonify(ok=True)

    def edge_by_idx(w, idx):
        return next((d for *_, d in w.G.edges(data=True) if d["idx"] == idx), None)

    @app.get("/api/conditions")
    def api_conditions():
        return jsonify(app.config["WORLD"].team)

    @app.post("/api/conditions/<int:idx>")
    def api_condition_set(idx):
        """The team's own check of one segment (form fields + an optional photo). Overrides every other source."""
        w = app.config["WORLD"]
        if edge_by_idx(w, idx) is None:
            return bad("unknown path segment", 404)
        f = request.form
        entry = {}
        for k, allowed in EDIT_CHOICES.items():
            v = f.get(k, "unknown")
            if v not in allowed:
                return bad(f"{k} must be one of: " + ", ".join(allowed))
            if v != "unknown":
                entry[k] = v
        if f.get("narrow") in ("1", "true", "on"):
            entry["narrow"] = True
        note = re.sub(r"\s+", " ", f.get("note", "")).strip()[:300]
        if note:
            entry["note"] = note
        contributor = f.get("contributor", "").strip().lower()
        if contributor and not re.fullmatch(r"c\d{2,3}", contributor):
            return bad("contributor id must look like c01 (no names, please)")
        photo, saved = request.files.get("photo"), None
        if photo and photo.filename:
            data = photo.read()
            if not any(data.startswith(m) for m in PHOTO_TYPES):
                return bad("the photo must be a JPEG or PNG")
            if len(data) > 15 * 1024 * 1024:
                return bad("the photo is too big (15 MB at most)")
            saved = f"seg{idx}-{uuid.uuid4().hex[:8]}.jpg"
            try:                                         # re-encode: drops the location and phone details in the file
                from PIL import Image, ImageOps
                im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
                im.thumbnail((1600, 1600))
                os.makedirs(w.p["photos"], exist_ok=True)
                im.save(os.path.join(w.p["photos"], saved), "JPEG", quality=85)
            except Exception:
                return bad("that photo could not be read")
        with w.lock:
            team = w._json("conditions", {})
            photos = team.get(str(idx), {}).get("photos", []) + ([saved] if saved else [])
            if not entry and not saved:
                return bad("set at least one thing about this path, or add a photo")
            if photos:
                entry["photos"] = photos
            entry.update(contributor=contributor, updated=time.strftime("%Y-%m-%d %H:%M"))
            team[str(idx)] = entry
            json.dump(team, open(w.p["conditions"], "w", encoding="utf-8"), indent=1)
            w.rebuild()
        e = next(x for x in w.view["edges"] if x["idx"] == idx)
        return jsonify(ok=True, condition=e["cond"], est=e["est"])

    @app.delete("/api/conditions/<int:idx>")
    def api_condition_clear(idx):
        """Forget the team's edit of a segment (and its photos); the other sources apply again."""
        w = app.config["WORLD"]
        with w.lock:
            team = w._json("conditions", {})
            old = team.pop(str(idx), None)
            if old is None:
                return bad("the team has not edited this path", 404)
            for ph in old.get("photos", []):
                fp = os.path.join(w.p["photos"], os.path.basename(ph))
                try:
                    os.remove(fp)
                except OSError:                          # already gone, or still being sent to a browser
                    pass
            json.dump(team, open(w.p["conditions"], "w", encoding="utf-8"), indent=1)
            w.rebuild()
        return jsonify(ok=True)

    @app.get("/photos/<name>")
    def photo(name):
        if not re.fullmatch(r"seg\d+-[0-9a-f]{8}\.jpg", name):
            abort(404)
        return send_from_directory(app.config["WORLD"].p["photos"], name, max_age=86400)

    @app.post("/api/upload")
    def api_upload():
        """Add one Sensor Logger export (.zip). The zip is read in place, never extracted."""
        w = app.config["WORLD"]
        f = request.files.get("file")
        contributor = request.form.get("contributor", "").strip().lower()
        carry = request.form.get("carry", "hand")
        if not f or not f.filename.lower().endswith(".zip"):
            return bad("choose the .zip that Sensor Logger exported")
        if not re.fullmatch(r"c\d{2,3}", contributor):
            return bad("contributor id must look like c01 (no names, please)")
        if carry not in ("hand", "pocket"):
            return bad("carry must be hand or pocket")
        up = os.path.join(w.p["raw"], "uploads"); os.makedirs(up, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(f.filename)[0])[:60]
        dest = os.path.join(up, f"{time.strftime('%Y%m%d-%H%M%S')}_{contributor}_{stem}.zip")
        f.save(dest)
        started = False
        try:
            if not zipfile.is_zipfile(dest):
                raise ValueError("this file is not a zip")
            names = {os.path.basename(n) for n in zipfile.ZipFile(dest).namelist()}
            missing = {"Accelerometer.csv", "Location.csv"} - names
            if missing or not ({"Gravity.csv", "TotalAcceleration.csv"} & names):
                need = sorted(missing | ({"Gravity.csv"} if not ({"Gravity.csv", "TotalAcceleration.csv"} & names) else set()))
                raise ValueError("missing " + ", ".join(need) + " (switch those sensors on in Sensor Logger)")
            json.dump(dict(contributor=contributor, carry=carry), open(dest + ".json", "w"))
            with w.lock:
                started = True
                before = w.real.get("summary", {}).get("n_walks", 0)
                summary = process(find_walks(w.p["raw"]), w.p["osm"], w.p["real"],
                                  load_contributors(os.path.join(w.p["raw"], "contributors.csv")),
                                  load_options(os.path.join(w.p["raw"], "walk_options.json")))
                if summary["n_walks"] <= before:
                    raise ValueError("the recording could not be read (no walking or no GPS found)")
                w.rebuild()
        except Exception as e:                           # bad upload: remove it so it can't break the next rebuild
            for p in (dest, dest + ".json"):
                if os.path.exists(p):
                    os.remove(p)
            if started:                                  # put the walks layer back as it was without this file
                with w.lock:
                    try:
                        process(find_walks(w.p["raw"]), w.p["osm"], w.p["real"],
                                load_contributors(os.path.join(w.p["raw"], "contributors.csv")),
                                load_options(os.path.join(w.p["raw"], "walk_options.json")))
                        w.rebuild()
                    except Exception:
                        pass
            return bad(f"could not use this recording: {e}")
        walk = next((x for x in w.real["walks"] if x["id"] == os.path.basename(dest)[:-4]), None)
        return jsonify(ok=True, summary=summary, walk=dict(title=walk["title"], length_m=walk["length_m"],
                       duration_s=walk["duration_s"], kerbs_marked=len(walk["labelled"]), stairs=len(walk["stairs"]),
                       jolts=len(walk["events"]), synced=walk["sync_ok"]) if walk else None)

    @app.errorhandler(413)
    def too_big(e):
        return bad("that file is too big (60 MB at most)", 413)

    @app.errorhandler(404)
    def not_found(e):
        if request.path.startswith(("/api/", "/photos/", "/static/")):
            return bad("not found", 404)
        return page()

    return app


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
        return ip
    except OSError:
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PathSense web app")
    ap.add_argument("--lan", action="store_true", help="also serve phones on the same Wi-Fi")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    app = create_app()
    host = "0.0.0.0" if a.lan else "127.0.0.1"
    print(f"\n  PathSense is running:  http://127.0.0.1:{a.port}")
    if a.lan and lan_ip():
        print(f"  On phones (same Wi-Fi): http://{lan_ip()}:{a.port}")
    print("  Stop it with Ctrl+C\n", flush=True)
    app.run(host=host, port=a.port, debug=a.debug, threaded=True)
