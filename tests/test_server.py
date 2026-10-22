"""Web app API on a tiny fixture campus: a 100 m path A-B with reported stairs, and a 320 m step-free detour A-C-D-B."""
import io, json, os, sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from server import create_app

NODES = [(0, 12.9700, 79.1600), (1, 12.9709, 79.1600), (2, 12.9700, 79.1610), (3, 12.9709, 79.1610)]
EDGES = [(0, 1, 100.0), (0, 2, 108.5), (2, 3, 100.0), (3, 1, 108.5)]
A, B = "12.9700,79.1600", "12.9709,79.1600"


@pytest.fixture
def client(tmp_path):
    flat = dict(grad=0.5, rough=0.1, kerbs=0, cls=0)
    nodes = [dict(id=i, x=(lon - 79.16) * 108600, y=(lat - 12.97) * 110600, lat=lat, lon=lon) for i, lat, lon in NODES]
    edges = [dict(u=u, v=v, idx=k, length=L, observed=False, name=f"path {k}", est=dict(flat), true=dict(flat),
                  geom=[[NODES[u][1], NODES[u][2]], [NODES[v][1], NODES[v][2]]], geom_from=u)
             for k, (u, v, L) in enumerate(EDGES)]
    (tmp_path / "outputs").mkdir(); (tmp_path / "data" / "ground_truth").mkdir(parents=True); (tmp_path / "data" / "raw").mkdir()
    json.dump(dict(nodes=nodes, edges=edges), open(tmp_path / "outputs" / "map.json", "w"))
    json.dump([dict(name="North gate", lat=12.9709, lon=79.16, kind="gate")], open(tmp_path / "data" / "osm_places.json", "w"))
    (tmp_path / "data" / "ground_truth" / "features.csv").write_text(
        "feature_id,type,lat,lon,landmark\nf001,steps,12.97045,79.1600,steps between A and B\n")
    app = create_app(root=str(tmp_path), auto_build=False)
    app.testing = True
    c = app.test_client(); c.root = tmp_path
    return c


def test_status_map_places(client):
    s = client.get("/api/status").get_json()
    assert s["ok"] and s["segments"] == 4 and s["barriers"] == 1 and s["places"] == 1
    m = client.get("/api/map").get_json()
    assert all("true" not in e for e in m["edges"])                      # simulator values stay hidden
    assert m["edges"][0]["reported"] and m["edges"][0]["est"]["stairs"] == 1
    assert set(m["profiles"]) >= {"manual_wheelchair", "shortest"}
    assert client.get("/api/places").get_json()[0]["name"] == "North gate"


def test_wheelchair_avoids_reported_stairs(client):
    r = client.get(f"/api/route?profile=manual_wheelchair&from={A}&to={B}").get_json()
    assert r["pathsense"]["stairs"] == 0 and r["pathsense"]["length"] > 300
    assert r["shortest"]["stairs"] == 1 and r["shortest"]["length"] == 100
    assert not r["same"] and not r["no_accessible_route"]
    assert r["shortest"]["issues"][0]["kind"] == "reported"
    s = client.get(f"/api/route?profile=shortest&src=0&dst=1").get_json()
    assert s["same"] and s["pathsense"]["length"] == 100


def test_route_errors(client):
    assert client.get(f"/api/route?profile=nope&from={A}&to={B}").status_code == 400
    assert client.get(f"/api/route?profile=crutches&from={A}&to={A}").status_code == 400
    assert client.get("/api/route?profile=crutches&src=0&dst=99").status_code == 400
    assert client.get("/api/route?profile=crutches").status_code == 400


def test_site_report_changes_route_and_can_be_removed(client):
    r = client.post("/api/reports", json=dict(type="blocked", lat=12.97045, lon=79.1610, note="  dug up  ", contributor="c07"))
    assert r.status_code == 200
    rep = r.get_json()["report"]; assert rep["note"] == "dug up"
    assert json.load(open(client.root / "data" / "reports.json"))[0]["id"] == rep["id"]
    # detour closed, direct path has stairs: no step-free way left
    q = client.get(f"/api/route?profile=manual_wheelchair&from={A}&to={B}").get_json()
    assert q["no_accessible_route"] and q["pathsense"]["stairs"] == 1
    assert client.get("/api/status").get_json()["barriers"] == 2
    assert client.delete(f"/api/reports/{rep['id']}").status_code == 200
    q = client.get(f"/api/route?profile=manual_wheelchair&from={A}&to={B}").get_json()
    assert not q["no_accessible_route"] and q["pathsense"]["stairs"] == 0


def test_report_validation(client):
    post = lambda **b: client.post("/api/reports", json=b)
    assert post(type="lava", lat=12.97045, lon=79.16).status_code == 400
    assert post(type="kerb").status_code == 400
    assert post(type="kerb", lat=12.99, lon=79.20).status_code == 400            # far from any path
    assert post(type="kerb", lat=12.97045, lon=79.16, contributor="George").status_code == 400
    assert client.delete("/api/reports/f001").status_code == 404                 # team reports can't be removed here


def test_upload_validation_leaves_nothing_behind(client):
    def up(name, data, contributor="c09"):
        return client.post("/api/upload", data=dict(file=(io.BytesIO(data), name), contributor=contributor),
                           content_type="multipart/form-data")
    assert up("walk.csv", b"x").status_code == 400
    assert up("walk.zip", b"not a zip").status_code == 400
    assert up("walk.zip", b"x", contributor="george").status_code == 400
    buf = io.BytesIO()
    import zipfile
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Accelerometer.csv", "time,x,y,z\n")
    r = up("walk.zip", buf.getvalue())
    assert r.status_code == 400 and "Location.csv" in r.get_json()["error"]
    left = client.root / "data" / "raw" / "uploads"
    assert not left.exists() or not any(left.iterdir())


def test_pages_fall_back_to_the_app(client):
    assert client.get("/").status_code == 200
    assert b"PathSense" in client.get("/walks").data
    assert client.get("/api/nothing").status_code == 404
    d = client.get("/api/demo").get_json()
    assert {"src", "dst", "from_label", "to_label", "profile"} <= set(d)


def test_team_path_edit_changes_route_and_photo_is_cleaned(client):
    # the detour's middle segment is dirt: the wheelchair now takes it only if the stairs path is worse
    r = client.post("/api/conditions/2", data=dict(surface="unpaved", kerb="unknown", steps="unknown", slope="unknown",
                                                  note="mud by the field", contributor="c01"), content_type="multipart/form-data")
    assert r.status_code == 200 and r.get_json()["condition"]["surface"] == "unpaved"
    m = client.get("/api/map").get_json()
    e = next(x for x in m["edges"] if x["idx"] == 2)
    assert e["est"]["cls"] == 2 and e["cond"]["surface_src"] == "team" and "mud" in e["cond"]["note"]
    assert client.get("/api/status").get_json()["coverage"]["team"] == 1
    # a photo with a fake GPS block is re-encoded (no EXIF left) and served
    from PIL import Image
    buf = io.BytesIO(); Image.new("RGB", (40, 30), "red").save(buf, "JPEG", exif=b"Exif\x00\x00MM\x00*\x00\x00\x00\x08\x00\x00")
    r = client.post("/api/conditions/2", data=dict(surface="unpaved", photo=(io.BytesIO(buf.getvalue()), "p.jpg")),
                    content_type="multipart/form-data")
    ph = r.get_json()["condition"]["photos"][0]
    img = client.get(f"/photos/{ph}")
    assert img.status_code == 200 and b"Exif" not in img.data[:200]
    img.close()
    assert client.post("/api/conditions/2", data=dict(photo=(io.BytesIO(b"GIF89a"), "x.gif")),
                       content_type="multipart/form-data").status_code == 400
    assert client.post("/api/conditions/2", data=dict(surface="lava"), content_type="multipart/form-data").status_code == 400
    assert client.post("/api/conditions/99", data=dict(surface="paved"), content_type="multipart/form-data").status_code == 404
    assert client.post("/api/conditions/1", data=dict(), content_type="multipart/form-data").status_code == 400
    assert client.get("/photos/map.json").status_code == 404 and client.get("/photos/seg1-..%2F..%2Fx.jpg").status_code == 404
    assert client.delete("/api/conditions/2").status_code == 200
    assert not (client.root / "data" / "photos" / ph).exists()
    assert client.delete("/api/conditions/2").status_code == 404


def test_team_ramp_and_kerb_marks(client):
    q = lambda: client.get(f"/api/route?profile=manual_wheelchair&from={A}&to={B}").get_json()
    client.post("/api/conditions/1", data=dict(kerb="kerb"), content_type="multipart/form-data")
    assert q()["pathsense"]["kerbs_found"] == 1 or q()["pathsense"]["length"] > 300
    client.post("/api/conditions/1", data=dict(kerb="ramp"), content_type="multipart/form-data")
    assert next(e for e in client.get("/api/map").get_json()["edges"] if e["idx"] == 1)["est"]["kerbs"] == 0


def test_button_pressed_kerbs_prefer_the_footpath():
    import networkx as nx
    from pathsense import conditions as C
    G = nx.Graph()
    G.add_node(0, lat=12.97, lon=79.16); G.add_node(1, lat=12.971, lon=79.16); G.add_node(2, lat=12.97, lon=79.16008); G.add_node(3, lat=12.971, lon=79.16008)
    G.add_edge(0, 1, idx=0, length=111, osm_highway="service", osm_surface="asphalt")      # road
    G.add_edge(2, 3, idx=1, length=111, osm_highway="footway")                             # footpath 8.7 m east of it
    G.add_edge(0, 2, idx=2, length=9, osm_highway="track")
    walks = [dict(id="w1", labelled=[dict(lat=12.9705, lon=79.16003, label="up")]),       # pressed closer to the road
             dict(id="w2", labelled=[dict(lat=12.97052, lon=79.16004, label="down")])]
    cond, spots, maybes = C.build(G, dict(walks=walks, clusters=[dict(lat=12.9702, lon=79.16, confirmed=True, n_walks=2)]))
    assert len(spots) == 1 and spots[0]["presses"] == 2 and spots[0]["edge"] == 1
    assert cond[1]["kerbs"] == 1 and cond[0]["kerbs"] == 0 and cond[0]["maybe"] == 1
    assert cond[0]["surface"] == "smooth" and cond[1]["surface"] == "unknown" and cond[2]["surface"] == "unpaved"
    assert C.attrs(cond[2])["poor"] == 1 and C.attrs(cond[1])["unknown"] == 1


def test_extra_link_and_vehicles_only(tmp_path, client):
    # a missing connection C->B with a kerb, and the direct stairs path marked vehicles-only
    json.dump([dict(name="link", **{"from": [12.9700, 79.1610], "to": [12.9709, 79.1600]}, surface="paved", kerbs=1)],
              open(client.root / "data" / "extra_links.json", "w"))
    json.dump({"0": dict(foot="no", note="cars only")}, open(client.root / "data" / "path_conditions.json", "w"))
    app = create_app(root=str(client.root), auto_build=False); c = app.test_client()
    m = c.get("/api/map").get_json()
    link = next(e for e in m["edges"] if e["name"] == "link")
    assert link["idx"] == 4 and link["est"]["kerbs"] == 1
    assert "not for walking (vehicles only)" in next(e for e in m["edges"] if e["idx"] == 0)["cond"]["lines"]
    r = c.get(f"/api/route?profile=shortest&from={A}&to={B}").get_json()
    assert r["pathsense"]["kerbs_found"] == 1 and r["pathsense"]["closed"] == 0     # walkers use the link, not the road
    # re-saving the segment in the editor keeps the vehicles-only mark
    c.post("/api/conditions/0", data=dict(surface="smooth"), content_type="multipart/form-data")
    assert json.load(open(client.root / "data" / "path_conditions.json"))["0"]["foot"] == "no"
