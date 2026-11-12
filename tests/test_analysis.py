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
