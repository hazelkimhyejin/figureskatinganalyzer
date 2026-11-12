"""Turns a PoseTrack into jumps, spins, speed and coaching feedback.

All distances are measured in "torso lengths" (TL: shoulder-centre to
hip-centre), which keeps the maths independent of zoom and camera distance.
TL is converted to metres with the skater's height when it is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from . import constants as C
from .pose import PoseTrack


# ----------------------------------------------------------------- helpers
def _interp(a: np.ndarray, valid: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill invalid rows of a (T, ...) array; gaps longer than max_gap stay NaN."""
    a = a.copy()
    T = len(a)
    ok = valid & np.all(np.isfinite(a.reshape(T, -1)), axis=1)
    if ok.sum() < 2:
        return a, ok
    idx = np.arange(T)
    flat = a.reshape(T, -1)
    for j in range(flat.shape[1]):
        flat[:, j] = np.interp(idx, idx[ok], flat[ok, j])
    filled = ok.copy()
    # re-mask long gaps
    bad = ~ok
    i = 0
    while i < T:
        if bad[i]:
            j = i
            while j < T and bad[j]:
                j += 1
            if (j - i) <= max_gap and i > 0 and j < T:
                filled[i:j] = True
            else:
                flat[i:j] = np.nan
            i = j
        else:
            i += 1
    return flat.reshape(a.shape), filled


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win) | 1)
    if win == 1 or len(x) < 3:
        return x.copy()
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    nan = ~np.isfinite(xp)
    xp0 = np.where(nan, 0, xp)
    k = np.ones(win)
    s = np.convolve(xp0, k, "valid")
    n = np.convolve((~nan).astype(float), k, "valid")
    out = s / np.maximum(n, 1)
    out[n == 0] = np.nan
    return out


def _rolling(x: np.ndarray, win: int, fn) -> np.ndarray:
    win = max(3, int(win) | 1)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(xp, win)
    with np.errstate(all="ignore"):
        return fn(view)


def _angle(a, b, c) -> np.ndarray:
    """Angle ABC in degrees for (T,2) arrays."""
    v1, v2 = a - b, c - b
    cos = np.sum(v1 * v2, -1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive-exclusive (start, end) runs of True."""
    out, i, T = [], 0, len(mask)
    while i < T:
        if mask[i]:
            j = i
            while j < T and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _nanmean(x, default=np.nan):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else default


def _nanmin(x, default=np.nan):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.min()) if len(x) else default


def _nanmax(x, default=np.nan):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.max()) if len(x) else default


# ----------------------------------------------------------------- signals
@dataclass
class Signals:
    fps: float
    t: np.ndarray
    tracked: np.ndarray
    P: np.ndarray            # (T,33,2) pixels
    vis: np.ndarray          # (T,33)
    TL: np.ndarray           # torso length px (rolling median)
    foot_rel: np.ndarray     # lowest-foot height above ice, TL
    hip_rel: np.ndarray      # hip height above its ground level, TL
    foot_base: np.ndarray    # px y of the ice line
    vx: np.ndarray           # skater velocity over the ice, TL/s
    vy: np.ndarray
    speed: np.ndarray        # TL/s
    yaw2: np.ndarray         # unwrapped doubled body yaw (rad); half-turn = 2*pi
    knee_L: np.ndarray
    knee_R: np.ndarray
    tilt: np.ndarray         # torso angle from vertical, deg
    wrist_dist: np.ndarray   # max wrist distance from shoulder centre, TL
    wrist_span: np.ndarray   # distance wrist-to-wrist, TL
    ankle_sep: np.ndarray    # TL
    sh_width: np.ndarray     # image shoulder width / TL
    toe_dir: np.ndarray      # (T,) x-direction the lower foot's toes point (+1/-1/0)
    face_x: np.ndarray       # world forward-vector x component


def compute_signals(tr: PoseTrack) -> Signals:
    fps = tr.fps
    T = len(tr)
    P, okP = _interp(tr.img[:, :, :2], tr.valid, max_gap=int(0.35 * fps))
    W, _ = _interp(tr.world, tr.valid, max_gap=int(0.35 * fps))
    vis = np.nan_to_num(tr.img[:, :, 2])

    sh = (P[:, C.L_SHOULDER] + P[:, C.R_SHOULDER]) / 2
    hp = (P[:, C.L_HIP] + P[:, C.R_HIP]) / 2
    torso = np.linalg.norm(sh - hp, axis=1)
    TL = _rolling(torso, int(2 * fps), lambda v: np.nanmedian(v, axis=1))
    TL = np.where(np.isfinite(TL) & (TL > 5), TL, np.nanmedian(torso) if np.isfinite(torso).any() else 50.0)

    foot_pts = P[:, [C.L_ANKLE, C.R_ANKLE, C.L_HEEL, C.R_HEEL, C.L_FOOT, C.R_FOOT], 1]
    foot_y = np.nanmax(foot_pts, axis=1) if T else np.zeros(0)
    foot_y = _smooth(foot_y, max(1, int(0.03 * fps)))
    foot_base = _rolling(foot_y, int(2.0 * fps), lambda v: np.nanpercentile(v, 85, axis=1))
    foot_base = _smooth(foot_base, int(0.5 * fps))
    foot_rel = (foot_base - foot_y) / TL

    hip_y = hp[:, 1]
    hip_base = _smooth(_rolling(hip_y, int(2.0 * fps), lambda v: np.nanpercentile(v, 70, axis=1)), int(0.5 * fps))
    hip_rel = (hip_base - hip_y) / TL

    # velocity over the ice: image motion minus background (camera pan) motion
    dh = np.vstack([np.zeros((1, 2)), np.diff(hp, axis=0)])
    v = (dh - tr.pan) * fps / TL[:, None]
    vx = _smooth(v[:, 0], int(0.3 * fps))
    vy = _smooth(v[:, 1], int(0.3 * fps))
    speed = np.hypot(vx, vy)

    # body yaw from world landmarks; the doubled angle makes it immune to
    # left/right label swaps (which happen when the skater's back is to camera)
    vs = W[:, C.R_SHOULDER] - W[:, C.L_SHOULDER]
    vh = W[:, C.R_HIP] - W[:, C.L_HIP]
    vv = vs + 0.5 * vh
    yaw = np.arctan2(vv[:, 2], vv[:, 0])
    y2 = 2 * yaw
    good = np.isfinite(y2)
    if good.sum() > 1:
        idx = np.arange(T)
        c, s = np.interp(idx, idx[good], np.cos(y2[good])), np.interp(idx, idx[good], np.sin(y2[good]))
        yaw2 = np.unwrap(np.arctan2(s, c))
    else:
        yaw2 = np.zeros(T)

    knee_L = _angle(P[:, C.L_HIP], P[:, C.L_KNEE], P[:, C.L_ANKLE])
    knee_R = _angle(P[:, C.R_HIP], P[:, C.R_KNEE], P[:, C.R_ANKLE])
    d = sh - hp
    tilt = np.degrees(np.arctan2(np.abs(d[:, 0]), np.abs(d[:, 1]) + 1e-9))
    wd = np.maximum(np.linalg.norm(P[:, C.L_WRIST] - sh, axis=1), np.linalg.norm(P[:, C.R_WRIST] - sh, axis=1)) / TL
    span = np.linalg.norm(P[:, C.L_WRIST] - P[:, C.R_WRIST], axis=1) / TL
    asep = np.linalg.norm(P[:, C.L_ANKLE] - P[:, C.R_ANKLE], axis=1) / TL
    shw = np.abs(P[:, C.R_SHOULDER, 0] - P[:, C.L_SHOULDER, 0]) / TL

    lower_is_L = P[:, C.L_ANKLE, 1] >= P[:, C.R_ANKLE, 1]
    toe_L = P[:, C.L_FOOT, 0] - P[:, C.L_HEEL, 0]
    toe_R = P[:, C.R_FOOT, 0] - P[:, C.R_HEEL, 0]
    toe = np.where(lower_is_L, toe_L, toe_R)
    toe_dir = np.where(np.abs(toe) > 0.08 * TL, np.sign(toe), 0.0)

    fwd = W[:, C.NOSE] - (W[:, C.L_SHOULDER] + W[:, C.R_SHOULDER]) / 2
    face_x = fwd[:, 0]

    return Signals(fps, tr.t, okP, P, vis, TL, foot_rel, hip_rel, foot_base, vx, vy, speed, yaw2,
                   knee_L, knee_R, tilt, wd, span, asep, shw, toe_dir, face_x)


# ----------------------------------------------------------------- results
@dataclass
class Issue:
    key: str
    title: str
    detail: str
    tip: str
    severity: int  # 1 minor, 2 moderate, 3 major


@dataclass
class Jump:
    n: int
    takeoff_i: int
    landing_i: int
    peak_i: int
    t_takeoff: float
    t_landing: float
    air_time: float
    height_m: float
    rev_measured: float
    rev_confidence: str
    jump_type: str
    type_confidence: float
    type_alternative: str
    type_reasons: list
    nominal_rev: float
    rotation_call: str        # "clean", "q", "<", "<<"
    code: str                 # e.g. "1A<"
    base_value: float | None
    rev_per_sec: float
    approach_speed: float     # TL/s
    takeoff_speed: float
    exit_speed: float
    distance_tl: float
    combo_with_prev: bool = False
    fall: bool = False
    issues: list = field(default_factory=list)
    strengths: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


@dataclass
class Spin:
    n: int
    start_i: int
    end_i: int
    t_start: float
    duration: float
    revolutions: float
    rev_per_sec: float
    travel_tl: float
    position: str
    issues: list = field(default_factory=list)
    strengths: list = field(default_factory=list)


# ----------------------------------------------------------------- jumps
def _nominal(rev: float, jtype: str) -> float:
    if jtype == "Waltz jump":
        return 0.5
    if jtype == "Axel":
        options = [1.5, 2.5, 3.5, 4.5]
    else:
        options = [1.0, 2.0, 3.0, 4.0]
    target = rev + 0.15  # measurements tend to read a little short
    return min(options, key=lambda o: abs(o - target))


def _rotation_call(short: float) -> str:
    if short >= C.DOWNGRADE_SHORT:
        return "<<"
    if short >= C.UR_SHORT:
        return "<"
    if short >= C.Q_SHORT:
        return "q"
    return "clean"


def _code(jtype: str, nominal: float, call: str) -> tuple[str, float | None]:
    """ISU-style element code and estimated base value."""
    abbr = C.JUMP_ABBR.get(jtype, "?")
    if jtype in ("Waltz jump", "Unknown"):
        return abbr, None
    lvl = int(round(nominal - 0.5)) if jtype == "Axel" else int(round(nominal))
    if call == "<<":
        if lvl <= 1:
            return f"{lvl}{abbr}<<", None
        lvl -= 1
    base = f"{lvl}{abbr}"
    bv = C.BASE_VALUES.get(base)
    if bv is not None and call == "<":
        bv = round(bv * C.UNDER_ROTATED_FACTOR, 2)
    return base + {"<": "<", "<<": "<<", "q": "q"}.get(call, ""), bv


def _classify(S: Signals, to: int, rev: float, air_dir: float, ccw_skater: bool) -> tuple[str, float, str, list]:
    fps = S.fps
    a = max(0, to - int(0.45 * fps))
    b = max(a + 1, to)
    reasons = []

    # forward or backward take-off?
    vxw = S.vx[a:b]
    moving = np.abs(vxw) > 0.4
    votes = []
    if moving.any():
        tvote = (np.sign(vxw) * S.toe_dir[a:b])[moving]
        tvote = tvote[tvote != 0]
        if len(tvote):
            votes.append(float(np.mean(tvote)))
        fx = S.face_x[a:b][moving]
        fv = np.sign(vxw[moving]) * np.sign(fx)
        fv = fv[np.isfinite(fv) & (np.abs(fx) > 0.02)]
        if len(fv):
            votes.append(float(np.mean(fv)))
    fwd_score = float(np.mean(votes)) if votes else 0.0

    if rev < 0.85 and fwd_score > 0.2:
        reasons.append("forward take-off with about half a turn")
        return "Waltz jump", 0.6, "Axel", reasons
    if fwd_score > 0.25:
        reasons.append(f"take-off looks forward (score {fwd_score:+.2f}) — only the Axel takes off forwards")
        return "Axel", min(0.95, 0.55 + fwd_score / 2), "Loop", reasons

    # backward take-off: toe-assisted or edge?
    w = slice(max(0, to - int(0.3 * fps)), max(1, to - int(0.02 * fps)))
    foot_y = S.P[w][:, [C.L_ANKLE, C.R_ANKLE], 1]
    near_ice = np.abs(foot_y - S.foot_base[w, None]) < 0.25 * S.TL[w, None]
    both_down = near_ice.all(axis=1)
    sep = S.ankle_sep[w]
    toe_frames = both_down & (sep > 0.85)
    toe_score = float(toe_frames.mean()) if len(toe_frames) else 0.0
    sep_early = _nanmax(S.ankle_sep[a:max(a + 1, to - int(0.15 * fps))], 0)
    sep_late = _nanmean(S.ankle_sep[max(0, to - int(0.1 * fps)):to + 1], 0)

    if toe_score > 0.15:
        reasons.append("both feet on the ice, wide apart just before take-off → toe pick")
        # which foot is picking? the one further behind the direction of travel
        dirn = np.sign(_nanmean(S.vx[a:b], 0)) or 1.0
        hipx = (S.P[w][:, C.L_HIP, 0] + S.P[w][:, C.R_HIP, 0]) / 2
        behind_L = _nanmean((S.P[w][:, C.L_ANKLE, 0] - hipx) * dirn)
        behind_R = _nanmean((S.P[w][:, C.R_ANKLE, 0] - hipx) * dirn)
        # moving backwards, "behind the direction of travel" is ahead of the body's back
        pick_left = behind_L > behind_R
        toe_loop_side = pick_left if ccw_skater else not pick_left
        if toe_loop_side:
            reasons.append(f"picking with the {'left' if pick_left else 'right'} toe → Toe Loop pattern")
            return "Toe Loop", 0.5 + 0.3 * min(1, toe_score * 2), "Flip", reasons
        # flip vs lutz: counter-rotation before take-off = Lutz
        pre = slice(max(0, to - int(0.6 * fps)), max(1, to - int(0.15 * fps)))
        pre_turn = (S.yaw2[pre][-1] - S.yaw2[pre][0]) if (pre.stop - pre.start) > 2 else 0.0
        counter = np.sign(pre_turn) != np.sign(air_dir) and abs(pre_turn) > 0.6
        reasons.append(f"picking with the {'left' if pick_left else 'right'} toe → Flip/Lutz pattern")
        if counter:
            reasons.append("upper body turning against the jump direction before take-off → Lutz (outside edge)")
            return "Lutz", 0.5, "Flip", reasons
        reasons.append("body already turning with the jump before take-off → Flip (inside edge)")
        return "Flip", 0.5, "Lutz", reasons

    # edge jumps
    if sep_early > 0.9 and sep_late < 0.7:
        reasons.append("free leg sweeps in from behind at take-off → Salchow")
        return "Salchow", 0.55, "Loop", reasons
    if sep_late < 0.6:
        reasons.append("legs close / crossed at take-off with no toe pick → Loop")
        return "Loop", 0.5, "Salchow", reasons
    reasons.append("backward edge take-off, pattern unclear")
    return "Salchow", 0.35, "Loop", reasons


def _jump_issues(S: Signals, j: Jump, to: int, ld: int, T: int):
    fps = S.fps
    issues, strengths = [], []
    air = slice(to, ld + 1)
    air_n = max(1, ld - to)

    # 1. take-off knee bend
    pre = slice(max(0, to - int(0.5 * fps)), to + 1)
    knee_to = _nanmin(np.minimum(S.knee_L[pre], S.knee_R[pre]))
    j.metrics["takeoff_knee_deg"] = round(knee_to, 1) if np.isfinite(knee_to) else None
    if np.isfinite(knee_to):
        if knee_to > 145:
            issues.append(Issue("knee_takeoff", "Shallow knee bend on take-off",
                                f"Deepest knee angle before take-off was {knee_to:.0f}° (aim for ~110–125°).",
                                "Sit deeper into the take-off edge and push up through the knee — that's where your height comes from.", 3))
        elif knee_to > 132:
            issues.append(Issue("knee_takeoff", "Knee bend could be deeper",
                                f"Deepest knee angle before take-off was {knee_to:.0f}°.",
                                "A little more bend on the take-off edge will give you more spring and air time.", 1))
        else:
            strengths.append(f"Good knee bend on take-off ({knee_to:.0f}°)")

    # 2. arm tuck timing
    wd = S.wrist_dist[air]
    tucked = np.where(wd < 0.95)[0]
    if len(tucked) == 0:
        issues.append(Issue("no_tuck", "Arms never fully pulled in",
                            "Wrists stayed away from the chest for the whole flight.",
                            "Snap the arms into the chest right after take-off — a tight tuck is the fastest way to rotate more.", 3))
        j.metrics["tuck_delay_pct"] = None
    else:
        frac = tucked[0] / air_n
        j.metrics["tuck_delay_pct"] = round(100 * frac)
        if frac > 0.35:
            issues.append(Issue("late_tuck", "Late arm pull-in",
                                f"Arms reached the tuck {100 * frac:.0f}% of the way through the flight.",
                                "Pull in faster off the ice — you're losing the first part of the flight for rotation.", 2))
        else:
            strengths.append(f"Quick pull-in ({100 * frac:.0f}% into the flight)")

    # 3. legs in the air
    mid = slice(to + int(0.2 * air_n), to + int(0.8 * air_n) + 1)
    sep_air = _nanmean(S.ankle_sep[mid])
    j.metrics["air_leg_gap_tl"] = round(sep_air, 2) if np.isfinite(sep_air) else None
    if np.isfinite(sep_air):
        if sep_air > 0.7:
            issues.append(Issue("open_legs", "Legs open in the air",
                                f"Feet were {sep_air:.1f} torso-lengths apart mid-flight.",
                                "Cross the legs tightly (back-spin position) — open legs act like brakes on rotation.", 2))
        else:
            strengths.append("Tight air position")

    # 4. axis tilt
    tilt_air = _nanmax(S.tilt[mid])
    j.metrics["air_tilt_deg"] = round(tilt_air, 1) if np.isfinite(tilt_air) else None
    if np.isfinite(tilt_air):
        if tilt_air > 30:
            issues.append(Issue("tilt", "Tilted rotation axis",
                                f"Body leaned up to {tilt_air:.0f}° off vertical in the air.",
                                "Keep the head level and shoulders square over the hips at take-off; a tilted axis makes landings risky.", 3))
        elif tilt_air > 20:
            issues.append(Issue("tilt", "Slight lean in the air",
                                f"Body leaned up to {tilt_air:.0f}° off vertical.",
                                "Stay tall through the core so the axis stays straight.", 1))
        else:
            strengths.append("Straight rotation axis")

    # 5. air time vs what this jump needs
    target = C.AIR_TIME_TARGET.get(j.nominal_rev)
    if target:
        need_rps = j.nominal_rev / max(j.air_time, 1e-3)
        j.metrics["needed_rev_per_s"] = round(need_rps, 2)
        if j.air_time < target - 0.05:
            issues.append(Issue("low_air", "Not enough air time",
                                f"{j.air_time:.2f}s in the air (≈{target:.2f}s is typical for this jump). "
                                f"At this height you'd need {need_rps:.1f} rev/s to finish the rotation; you rotated at {j.rev_per_sec:.1f}.",
                                "Height comes from speed + knee bend + timing of the arm/free-leg swing at take-off.", 2))
        else:
            strengths.append(f"Good air time ({j.air_time:.2f}s)")

    # 6. rotation
    short = j.nominal_rev - j.rev_measured
    if j.rotation_call != "clean":
        label = {"q": "Quarter short (q)", "<": "Under-rotated (<)", "<<": "Downgraded (<<)"}[j.rotation_call]
        issues.append(Issue("rotation", label,
                            f"Measured about {j.rev_measured:.2f} of {j.nominal_rev:g} revolutions ({short:.2f} short).",
                            "Finish the rotation in the air: faster tuck, more height, and don't open up early looking for the landing.",
                            3 if j.rotation_call in ("<", "<<") else 2))
    else:
        strengths.append(f"Rotation looks complete ({j.rev_measured:.2f} rev)")

    # 7. speed into and out of the jump
    if j.approach_speed > 0.3:
        if j.takeoff_speed < 0.65 * j.approach_speed:
            issues.append(Issue("stall", "Losing speed into the take-off",
                                f"Speed dropped from {j.approach_speed:.1f} to {j.takeoff_speed:.1f} TL/s right before take-off.",
                                "Keep the glide running through the take-off — hesitating or scraping on the entry edge kills height.", 2))
        if j.exit_speed < 0.55 * j.approach_speed and not j.fall:
            issues.append(Issue("flow_out", "Landing loses flow",
                                f"Exit speed was {j.exit_speed:.1f} vs {j.approach_speed:.1f} TL/s on the approach.",
                                "Land on a soft knee and push the free leg back to run out on a clean back outside edge.", 2))
        elif not j.fall:
            strengths.append("Keeps speed out of the landing")
    if j.approach_speed < 1.5:
        issues.append(Issue("slow_entry", "Slow approach",
                            f"Approach speed was only {j.approach_speed:.1f} torso-lengths per second.",
                            "Build more speed with stronger crossovers into the entry — speed converts into height and distance.", 1))

    # 8. landing knee
    post = slice(ld, min(T, ld + int(0.3 * fps)))
    knee_ld = _nanmin(np.minimum(S.knee_L[post], S.knee_R[post]))
    j.metrics["landing_knee_deg"] = round(knee_ld, 1) if np.isfinite(knee_ld) else None
    if np.isfinite(knee_ld) and not j.fall:
        if knee_ld > 155:
            issues.append(Issue("stiff_landing", "Stiff landing",
                                f"Landing knee only bent to {knee_ld:.0f}°.",
                                "Absorb the landing through the knee and ankle (aim for ~120–140°) — it protects you and keeps flow.", 2))
        else:
            strengths.append("Soft, absorbed landing")

    # 9. check-out / landing position
    co = slice(min(T - 1, ld + int(0.1 * fps)), min(T, ld + int(0.7 * fps)))
    sep_out = _nanmax(S.ankle_sep[co])
    span_out = _nanmax(S.wrist_span[co])
    j.metrics["landing_free_leg_tl"] = round(sep_out, 2) if np.isfinite(sep_out) else None
    if np.isfinite(sep_out) and not j.fall:
        if sep_out < 1.0 or (np.isfinite(span_out) and span_out < 1.8):
            issues.append(Issue("checkout", "Weak landing position",
                                "Free leg and/or arms didn't extend after landing.",
                                "Check out: free leg back and extended, arms open to stop the rotation and hold the edge.", 1))
        else:
            strengths.append("Strong landing position")

    # 10. pre-rotation on the ice
    pr = slice(max(0, to - int(0.12 * fps)), to + 1)
    if pr.stop - pr.start > 2:
        pre_rot = abs(S.yaw2[pr][-1] - S.yaw2[pr][0]) / (4 * np.pi)
        j.metrics["prerotation_rev"] = round(float(pre_rot), 2)
        if pre_rot > 0.35 and j.jump_type != "Axel":
            issues.append(Issue("prerotate", "Possible pre-rotation on take-off",
                                f"About {pre_rot:.2f} rev of turning happened on the ice in the last 0.1s.",
                                "Let the take-off edge carry you up before you rotate; check the slow-mo clip to confirm.", 1))

    if j.fall:
        issues.append(Issue("fall", "Fall or hand down on landing",
                            "Body dropped / touched the ice right after landing.",
                            "Usually a result of under-rotation or a tilted axis — fix those first.", 3))
    return issues, strengths


def detect_jumps(S: Signals, ccw_skater: bool = True, hint: str | None = None) -> list[Jump]:
    fps, T = S.fps, len(S.t)
    if T < 5:
        return []
    fr = np.nan_to_num(S.foot_rel, nan=0)
    air_mask = (fr > 0.10) & S.tracked
    segs = _segments(air_mask)
    # merge tiny gaps
    merged = []
    for s, e in segs:
        if merged and s - merged[-1][1] <= max(1, int(0.04 * fps)):
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    jumps: list[Jump] = []
    for s, e in merged:
        dur = (e - s) / fps
        if not (0.15 <= dur <= 1.3):
            continue
        if fr[s:e].max() < 0.22 or _nanmax(S.hip_rel[s:e], 0) < 0.12:
            continue
        to = s
        while to > 0 and fr[to - 1] > 0.03 and s - to < int(0.15 * fps):
            to -= 1
        ld = e - 1
        while ld < T - 1 and fr[ld + 1] > 0.03 and ld - e < int(0.15 * fps):
            ld += 1
        air_time = (ld - to + 1) / fps
        peak = to + int(np.argmax(fr[to:ld + 1]))

        d2 = S.yaw2[ld] - S.yaw2[to]
        rev = abs(d2) / (4 * np.pi)
        # cross-check with shoulder-width minima (2 per revolution)
        w = _smooth(S.sh_width[to:ld + 1], 3)
        mins = int(np.sum((w[1:-1] < w[:-2]) & (w[1:-1] < w[2:]) & (w[1:-1] < 0.6))) if len(w) > 3 else 0
        rev_w = mins / 2
        agree = abs(rev - rev_w) <= 0.5 or mins == 0
        frames_per_rev = fps / max(rev / max(air_time, 1e-3), 1e-3)
        conf = "high" if agree and frames_per_rev >= 12 else ("medium" if agree or frames_per_rev >= 12 else "low")
        if not agree and rev_w > 0 and conf == "low":
            rev = (rev + rev_w) / 2

        jtype, tconf, alt, reasons = _classify(S, to, rev, np.sign(d2), ccw_skater)
        if hint and hint != "Auto-detect" and tconf < 0.65 and hint in (jtype, alt):
            reasons.append(f"you said you're practising the {hint}, which fits the evidence")
            if hint != jtype:
                jtype, alt = hint, jtype
            tconf = max(tconf, 0.65)

        nominal = _nominal(rev, jtype)
        call = _rotation_call(nominal - rev)
        code, bv = _code(jtype, nominal, call)

        a0 = max(0, to - int(1.0 * fps)); a1 = max(a0 + 1, to - int(0.3 * fps))
        approach = _nanmean(S.speed[a0:a1], 0)
        tko = _nanmean(S.speed[max(0, to - int(0.2 * fps)):to + 1], 0)
        ex = _nanmean(S.speed[min(T - 1, ld + int(0.2 * fps)):min(T, ld + int(0.7 * fps))], 0)
        dist = float(np.nansum(S.speed[to:ld + 1]) / fps)

        # fall / hand-down detection
        after = slice(ld, min(T, ld + int(1.0 * fps)))
        wrist_low = np.minimum(np.abs(S.P[after][:, C.L_WRIST, 1] - S.foot_base[after]),
                               np.abs(S.P[after][:, C.R_WRIST, 1] - S.foot_base[after])) / S.TL[after]
        fall = bool((_nanmin(S.hip_rel[after], 0) < -1.3) or (_nanmax(S.tilt[after], 0) > 60)
                    or (_nanmin(wrist_low, 9) < 0.35))

        j = Jump(
            n=len(jumps) + 1, takeoff_i=to, landing_i=ld, peak_i=peak,
            t_takeoff=float(S.t[to]), t_landing=float(S.t[ld]), air_time=round(air_time, 3),
            height_m=round(C.G * air_time ** 2 / 8, 3), rev_measured=round(float(rev), 2), rev_confidence=conf,
            jump_type=jtype, type_confidence=round(tconf, 2), type_alternative=alt, type_reasons=reasons,
            nominal_rev=nominal, rotation_call=call, code=code, base_value=bv,
            rev_per_sec=round(float(rev / max(air_time, 1e-3)), 2),
            approach_speed=round(approach, 2), takeoff_speed=round(tko, 2), exit_speed=round(ex, 2),
            distance_tl=round(dist, 2), fall=fall,
        )
        if jumps and (S.t[to] - jumps[-1].t_landing) < 0.7:
            j.combo_with_prev = True
        j.issues, j.strengths = _jump_issues(S, j, to, ld, T)
        jumps.append(j)
    return jumps


def regrade_jump(S: Signals, j: Jump, new_type: str) -> Jump:
    """Re-score a jump after the user corrects its type."""
    j.jump_type = new_type
    j.type_confidence = 1.0
    j.type_reasons = j.type_reasons + ["type set by you"]
    j.nominal_rev = _nominal(j.rev_measured, new_type)
    j.rotation_call = _rotation_call(j.nominal_rev - j.rev_measured)
    j.code, j.base_value = _code(new_type, j.nominal_rev, j.rotation_call)
    j.issues, j.strengths = _jump_issues(S, j, j.takeoff_i, j.landing_i, len(S.t))
    return j


# ----------------------------------------------------------------- spins
def detect_spins(S: Signals, jumps: list[Jump]) -> list[Spin]:
    fps, T = S.fps, len(S.t)
    if T < int(fps):
        return []
    rate = np.abs(np.gradient(S.yaw2)) * fps / (4 * np.pi)  # rev/s
    rate = _smooth(rate, int(0.4 * fps))
    in_jump = np.zeros(T, bool)
    for j in jumps:
        in_jump[max(0, j.takeoff_i - int(0.2 * fps)):min(T, j.landing_i + int(0.3 * fps))] = True
    mask = (rate > 1.0) & (np.nan_to_num(S.foot_rel) < 0.15) & ~in_jump & S.tracked
    spins = []
    for s, e in _segments(mask):
        dur = (e - s) / fps
        if dur < 1.2:
            continue
        revs = abs(S.yaw2[e - 1] - S.yaw2[s]) / (4 * np.pi)
        if revs < 2:
            continue
        travel = float(np.nansum(S.speed[s:e]) / fps)
        tilt = _nanmean(S.tilt[s:e], 0)
        knee = _nanmean(np.minimum(S.knee_L[s:e], S.knee_R[s:e]), 180)
        if tilt > 55:
            pos = "Camel"
        elif knee < 115:
            pos = "Sit"
        else:
            pos = "Upright"
        sp = Spin(len(spins) + 1, s, e - 1, float(S.t[s]), round(dur, 2), round(float(revs), 1),
                  round(float(revs / dur), 2), round(travel, 2), pos)
        if travel / max(revs, 1) > 0.35:
            sp.issues.append(Issue("travel", "Travelling spin",
                                   f"Drifted about {travel:.1f} torso-lengths over {revs:.0f} revolutions.",
                                   "Centre the spin: enter on a controlled edge and get over the ball of the foot quickly.", 2))
        else:
            sp.strengths.append("Well centred")
        if revs / dur < 1.8:
            sp.issues.append(Issue("slow_spin", "Slow rotation",
                                   f"{revs / dur:.1f} rev/s.",
                                   "Pull arms and free leg in tighter and lift through the core to speed up.", 1))
        else:
            sp.strengths.append(f"Good speed ({revs / dur:.1f} rev/s)")
        if revs < 3:
            sp.issues.append(Issue("few_revs", "Too few revolutions",
                                   f"Only {revs:.1f} revolutions.",
                                   "Hold each position for at least 3 revolutions so it counts toward the level.", 2))
        spins.append(sp)
    return spins


# ----------------------------------------------------------------- report
def analyze(tr: PoseTrack, ccw_skater: bool = True, hint: str | None = None,
            height_m: float | None = None) -> dict:
    S = compute_signals(tr)
    jumps = detect_jumps(S, ccw_skater, hint)
    spins = detect_spins(S, jumps)
    return build_report(S, jumps, spins, height_m)


def build_report(S: Signals, jumps: list[Jump], spins: list[Spin], height_m: float | None) -> dict:
    tl_m = (height_m * C.TORSO_TO_HEIGHT) if height_m else None

    # aggregate issues across jumps & spins
    tally: dict[str, dict] = {}
    for item in list(jumps) + list(spins):
        for iss in item.issues:
            d = tally.setdefault(iss.key, {"title": iss.title, "tip": iss.tip, "count": 0, "score": 0})
            d["count"] += 1
            d["score"] += iss.severity
    priorities = sorted(tally.values(), key=lambda d: (-d["score"], -d["count"]))[:5]

    strengths: dict[str, int] = {}
    for item in list(jumps) + list(spins):
        for s in item.strengths:
            key = s.split(" (")[0]
            strengths[key] = strengths.get(key, 0) + 1
    top_strengths = sorted(strengths.items(), key=lambda kv: -kv[1])[:5]

    moving = S.speed[np.isfinite(S.speed)]
    clean = sum(1 for j in jumps if j.rotation_call == "clean" and not j.fall)
    return {
        "signals": S,
        "jumps": jumps,
        "spins": spins,
        "tl_m": tl_m,
        "summary": {
            "duration_s": float(S.t[-1] - S.t[0]) if len(S.t) else 0.0,
            "tracked_pct": float(100 * S.tracked.mean()) if len(S.t) else 0.0,
            "n_jumps": len(jumps),
            "n_clean": clean,
            "n_falls": sum(j.fall for j in jumps),
            "n_spins": len(spins),
            "avg_air": _nanmean([j.air_time for j in jumps], 0),
            "max_air": _nanmax([j.air_time for j in jumps], 0),
            "avg_speed_tl": float(np.median(moving)) if len(moving) else 0.0,
            "max_speed_tl": float(np.percentile(moving, 95)) if len(moving) else 0.0,
            "total_bv": round(sum(j.base_value or 0 for j in jumps), 2),
        },
        "priorities": priorities,
        "strengths": top_strengths,
    }


def to_jsonable(report: dict) -> dict:
    out = {k: v for k, v in report.items() if k != "signals"}
    out["jumps"] = [asdict(j) for j in report["jumps"]]
    out["spins"] = [asdict(s) for s in report["spins"]]
    return out
