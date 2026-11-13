import json
import numpy as np
import pytest

from skate.analysis import analyze, to_jsonable, compute_signals, regrade_jump
from tests.synth import program, render


@pytest.mark.parametrize("fps", [30, 60])
def test_clean_program(fps):
    r = analyze(render(program(fps=fps), fps))
    js = r["jumps"]
    assert len(js) == 2
    assert js[0].jump_type == "Axel" and js[0].code == "1A"
    assert abs(js[0].rev_measured - 1.5) < 0.2
    assert js[1].jump_type == "Toe Loop" and abs(js[1].rev_measured - 1.0) < 0.2
    assert len(r["spins"]) == 1 and abs(r["spins"][0].revolutions - 7.5) < 0.6
    json.dumps(to_jsonable(r), default=float)


def test_under_rotated_axel():
    r = analyze(render(program(fps=60, axel_rev=1.12, spin=False), 60))
    a = r["jumps"][0]
    assert a.jump_type == "Axel"
    assert a.rotation_call in ("<", "<<")
    assert any(i.key == "rotation" for i in a.issues)


def test_camera_pan_is_compensated():
    fps = 60
    tr = render(program(fps=fps, spin=False), fps)
    # simulate a camera that tracks the skater: subtract hip motion from image, record as pan
    hip = (tr.img[:, 23, 0] + tr.img[:, 24, 0]) / 2
    shift = hip - hip[0]
    tr.img[:, :, 0] -= shift[:, None]
    tr.pan[:, 0] = -np.diff(np.r_[shift[0], shift])  # background moves opposite to the skater
    r = analyze(tr)
    assert len(r["jumps"]) == 2
    assert r["jumps"][0].jump_type == "Axel"
    assert r["jumps"][0].approach_speed > 4


def test_regrade():
    tr = render(program(fps=60, spin=False), 60)
    r = analyze(tr)
    S = compute_signals(tr)
    j = regrade_jump(S, r["jumps"][1], "Flip")
    assert j.code.startswith("1F")


def test_no_jumps_when_just_gliding():
    from tests.synth import FACE_RIGHT
    F = [dict(x=0.5 + 3 * i / 60, lift=0, yaw=FACE_RIGHT) for i in range(300)]
    r = analyze(render(F, 60))
    assert r["jumps"] == [] and r["spins"] == []


# ------------------------------------------------------------- triples and quads
from tests.synth import elements_program  # noqa: E402

HIGH_LEVEL = [("2S", "salchow", 2, 0.50), ("3S", "salchow", 3, 0.62), ("2Lo", "loop", 2, 0.52),
              ("3Lo", "loop", 3, 0.63), ("3T", "toe_loop", 3, 0.60), ("4T", "toe_loop", 4, 0.70),
              ("3F", "flip", 3, 0.62), ("3Lz", "lutz", 3, 0.64), ("2A", "axel", 2.5, 0.58),
              ("3A", "axel", 3.5, 0.68), ("4A", "axel", 4.5, 0.76), ("4Lz", "lutz", 4, 0.72)]


@pytest.mark.parametrize("fps", [30, 60, 120, 240])
def test_triples_and_quads(fps):
    r = analyze(render(elements_program([c[1:] for c in HIGH_LEVEL], fps=fps), fps, seed=fps))
    assert [j.code for j in r["jumps"]] == [c[0] for c in HIGH_LEVEL]


@pytest.mark.parametrize("fps", [60, 120])
def test_under_rotation_calls_high_level(fps):
    cases = [("3A", "axel", 3.42, 0.68), ("3A<", "axel", 3.12, 0.66), ("3A<<", "axel", 2.9, 0.66),
             ("4A<", "axel", 4.12, 0.74), ("3Sq", "salchow", 2.76, 0.62), ("4T<", "toe_loop", 3.62, 0.70)]
    r = analyze(render(elements_program([c[1:] for c in cases], fps=fps), fps, noise=0.012, world_noise=0.03))
    assert [j.code for j in r["jumps"]] == [c[0] for c in cases]
    downgraded = r["jumps"][2]
    assert downgraded.base_value == 3.30  # 3A<< is valued as a 2A


def test_attempted_level_resolves_ambiguity():
    # 2.85 rev is equally close to an over-rotated 2A and a downgraded 3A
    F = elements_program([("axel", 2.85, 0.66)], fps=30)
    auto = analyze(render(F, 30))["jumps"][0]
    told = analyze(render(F, 30), level=3)["jumps"][0]
    assert told.code == "3A<<"
    assert auto.code in ("2A", "3A<<")


def test_quad_axel_at_30fps_is_not_aliased():
    r = analyze(render(elements_program([("axel", 4.5, 0.70)], fps=30), 30))
    assert r["jumps"][0].code == "4A" and abs(r["jumps"][0].rev_measured - 4.5) < 0.15
