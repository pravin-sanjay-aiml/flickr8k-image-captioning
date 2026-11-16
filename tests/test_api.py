import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
from backend.app import app
c = TestClient(app)

def test_health(): r = c.get("/health").json(); assert r["status"] == "ok" and r["vocab"] > 1000
def test_samples(): s = c.get("/samples").json(); assert len(s) == 24 and c.get(s[0]["image"]).status_code == 200
def test_caption_beam_greedy():
    for m in ("beam", "greedy"):
        r = c.get(f"/caption/sample/3?mode={m}").json(); assert len(r["caption"].split()) >= 3 and len(r["references"]) == 5
def test_dog_sample_sane():  # sample 0..: just checks beam output is deterministic
    assert c.get("/caption/sample/1").json()["caption"] == c.get("/caption/sample/1").json()["caption"]
def test_bad_inputs(): assert c.get("/caption/sample/99").status_code == 404; assert c.get("/caption/sample/0?mode=x").status_code == 422
def test_upload_disabled_without_weights():
    os.environ.pop("CLIP_WEIGHTS", None); r = c.post("/caption/upload", files={"file": ("a.jpg", b"x", "image/jpeg")}); assert r.status_code == 503
def test_index(): assert "Image captioning demo" in c.get("/").text
