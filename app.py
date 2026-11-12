"""Skate Analyzer — upload a figure-skating video, get jump-by-jump feedback."""
from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from skate import constants as C
from skate.analysis import analyze, build_report, regrade_jump, to_jsonable
from skate.pose import extract_poses, video_info
from skate.render import jump_clip, keyframe_strip

st.set_page_config(page_title="Skate Analyzer", page_icon="⛸️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.issue {border-left: 4px solid #e8a33d; padding: .45rem .8rem; margin: .35rem 0; background: rgba(232,163,61,.08); border-radius: 4px;}
.issue.sev3 {border-color: #e5484d; background: rgba(229,72,77,.08);}
.issue.sev1 {border-color: #8e8ea0; background: rgba(142,142,160,.08);}
.good {border-left: 4px solid #30a46c; padding: .3rem .8rem; margin: .25rem 0; background: rgba(48,164,108,.08); border-radius: 4px;}
.code {font-size: 2.1rem; font-weight: 700; letter-spacing: .02em;}
.small {opacity: .7; font-size: .85rem;}
</style>
""", unsafe_allow_html=True)

SEV = {3: "Major", 2: "Moderate", 1: "Minor"}

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    height_cm = st.number_input("Skater height (cm)", 100, 220, 160,
                                help="Used to turn body proportions into real speeds (km/h) and distances.")
    direction = st.radio("Jump rotation direction", ["Counter-clockwise (most skaters)", "Clockwise"],
                         help="Needed to tell Toe Loop apart from Flip/Lutz (which toe picks).")
    hint = st.selectbox("Jump you're practising (optional)", ["Auto-detect"] + C.JUMP_TYPES[:-1],
                        help="Used as a tie-breaker when the take-off is ambiguous.")
    with st.expander("Advanced"):
        variant = st.select_slider("Pose model", ["lite", "full", "heavy"], value="full",
                                   help="lite = fastest, heavy = most accurate (slow on long videos).")
        max_fps = st.select_slider("Max analysis fps", [30, 60, 120, 240], value=60,
                                   help="Higher keeps more frames from slow-mo videos (better rotation counts, slower).")
        max_width = st.select_slider("Processing width (px)", [480, 640, 960], value=640)
    st.divider()
    st.markdown("""**🎥 Filming tips**
- Film from the side, whole body in frame
- Phone **slow-mo (120/240 fps)** gives the best rotation counts
- Steady camera, follow the skater smoothly
- One skater in the shot""")

ccw = direction.startswith("Counter")
tl_m = height_cm / 100 * C.TORSO_TO_HEIGHT


def speed_txt(tl_s: float) -> str:
    return f"{tl_s * tl_m * 3.6:.1f} km/h"


# ---------------------------------------------------------------- header + upload
st.title("⛸️ Skate Analyzer")
st.caption("Drop a skating video. The app tracks your body, finds every jump and spin, names the jump, "
           "counts the rotation, measures air time, height and speed, and tells you what to fix.")

up = st.file_uploader("Upload a video (any length)", type=["mp4", "mov", "m4v", "avi", "webm", "mkv"])

if up is None and os.environ.get("SKATE_DEV_VIDEO"):  # local testing hook
    class _LocalFile:
        name = os.path.basename(os.environ["SKATE_DEV_VIDEO"])

        def getvalue(self):
            with open(os.environ["SKATE_DEV_VIDEO"], "rb") as fh:
                return fh.read()
    up = _LocalFile()

if up is None:
    c1, c2, c3 = st.columns(3)
    c1.markdown("**🔍 Detects**\n\nAxel, Toe Loop, Salchow, Loop, Flip, Lutz, Waltz jump, combinations and spins")
    c2.markdown("**📏 Measures**\n\nRotations, air time, jump height, approach / take-off / exit speed, knee bend, arm tuck timing")
    c3.markdown("**🎯 Tells you**\n\nUnder-rotation calls (q, <, <<), falls, top fixes with coaching tips, slow-mo clips")
    st.stop()

data = up.getvalue()
vid_hash = hashlib.sha1(data[:2_000_000] + str(len(data)).encode()).hexdigest()[:16]
if st.session_state.get("vid_hash") != vid_hash:
    for k in ("track", "report", "track_key", "clips"):
        st.session_state.pop(k, None)
    path = os.path.join(tempfile.gettempdir(), f"skate_{vid_hash}{os.path.splitext(up.name)[1] or '.mp4'}")
    with open(path, "wb") as fh:
        fh.write(data)
    st.session_state.vid_hash = vid_hash
    st.session_state.vid_path = path
path = st.session_state.vid_path
info = video_info(path)
del data

if info["frames"] == 0 or info["width"] == 0:
    st.error("Couldn't read this video. Try exporting it as MP4 (H.264).")
    st.stop()

dur = info["duration"]
col_v, col_s = st.columns([2, 1])
with col_v:
    st.video(path)
with col_s:
    st.markdown(f"**{up.name}**")
    st.markdown(f"{info['width']}×{info['height']} · {info['fps']:.0f} fps · {dur:.1f}s")
    if info["fps"] < 50:
        st.info("Tip: at under 50 fps, fast rotations are only sampled a few times — counts are less precise. "
                "Phone slow-mo works best.", icon="ℹ️")
    rng = st.slider("Analyse this part (seconds)", 0.0, max(dur, 0.1), (0.0, max(dur, 0.1)), step=0.5)
    est = (rng[1] - rng[0]) * min(info["fps"], max_fps) * {"lite": 0.02, "full": 0.035, "heavy": 0.09}[variant]
    st.caption(f"Estimated processing time: ~{max(est, 3) / 60:.0f}–{max(est, 3) * 2 / 60:.0f} min" if est > 60
               else "Estimated processing time: under a minute or two")
    go_btn = st.button("Analyse video", type="primary", width="stretch")

track_key = (vid_hash, variant, max_fps, max_width, rng)
if go_btn and st.session_state.get("track_key") != track_key:
    bar = st.progress(0.0, "Loading pose model…")

    def cb(p, eta):
        bar.progress(p, f"Tracking your body… {p * 100:.0f}% (about {max(eta, 0):.0f}s left)")

    try:
        tr = extract_poses(path, variant=variant, max_width=max_width, start_s=rng[0],
                           end_s=rng[1] if rng[1] < dur - 0.05 else None, max_fps=max_fps, progress_cb=cb)
    except Exception as e:  # noqa: BLE001
        bar.empty()
        st.error(f"Pose tracking failed: {e}")
        st.stop()
    bar.empty()
    st.session_state.track = tr
    st.session_state.track_key = track_key
    st.session_state.pop("report", None)
    st.session_state.clips = {}

if "track" not in st.session_state:
    st.stop()

tr = st.session_state.track
rep_key = (track_key, ccw, hint)
if st.session_state.get("report_key") != rep_key or "report" not in st.session_state:
    st.session_state.report = analyze(tr, ccw_skater=ccw, hint=hint, height_m=height_cm / 100)
    st.session_state.report_key = rep_key
R = st.session_state.report
S = R["signals"]
jumps, spins, sm = R["jumps"], R["spins"], R["summary"]

if sm["tracked_pct"] < 40:
    st.warning(f"Only found the skater in {sm['tracked_pct']:.0f}% of frames — results may be incomplete. "
               "Make sure the whole body is visible and the skater isn't too small in the frame.")

# ---------------------------------------------------------------- tabs
t_over, t_jumps, t_spins, t_speed, t_export = st.tabs(["📋 Overview", "🦘 Jumps", "🌀 Spins", "⚡ Speed", "⬇️ Export"])

with t_over:
    m = st.columns(6)
    m[0].metric("Jumps", sm["n_jumps"])
    m[1].metric("Fully rotated", f"{sm['n_clean']}/{sm['n_jumps']}" if sm["n_jumps"] else "–")
    m[2].metric("Falls", sm["n_falls"])
    m[3].metric("Best air time", f"{sm['max_air']:.2f}s" if sm["n_jumps"] else "–")
    m[4].metric("Typical speed", speed_txt(sm["avg_speed_tl"]))
    m[5].metric("Est. base value", f"{sm['total_bv']:.2f}")

    if not jumps and not spins:
        st.info("No jumps or spins found. Check that the feet are visible when the skater lands, "
                "or try the 'heavy' pose model in Advanced settings.")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("🎯 What to work on")
        if R["priorities"]:
            for i, p in enumerate(R["priorities"], 1):
                st.markdown(f"**{i}. {p['title']}** — seen {p['count']}×  \n{p['tip']}")
        else:
            st.markdown("Nothing major flagged — nice skating!")
    with c2:
        st.subheader("💪 What's working")
        for s, n in R["strengths"]:
            st.markdown(f"<div class='good'>{s} <span class='small'>({n}×)</span></div>", unsafe_allow_html=True)

    if jumps:
        st.subheader("Element list")
        rows = []
        for j in jumps:
            rows.append({
                "#": j.n, "Time": f"{j.t_takeoff:.1f}s", "Element": ("+" if j.combo_with_prev else "") + j.code,
                "Jump": j.jump_type, "Rotations": j.rev_measured, "Air (s)": j.air_time,
                "Height (cm)": round(j.height_m * 100), "Approach": speed_txt(j.approach_speed),
                "Exit": speed_txt(j.exit_speed), "Fall": "yes" if j.fall else "",
                "Est. BV": j.base_value if j.base_value is not None else "–",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # timeline
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S.t, y=np.nan_to_num(S.foot_rel) * tl_m * 100, name="Foot height (cm)",
                             line=dict(color="#8e6ff7", width=1.5)))
    for j in jumps:
        fig.add_vrect(x0=j.t_takeoff, x1=j.t_landing, fillcolor="#8e6ff7", opacity=0.18, line_width=0,
                      annotation_text=j.code, annotation_position="top left")
    for s in spins:
        fig.add_vrect(x0=s.t_start, x1=s.t_start + s.duration, fillcolor="#4cc3d9", opacity=0.15, line_width=0,
                      annotation_text=f"{s.position} spin", annotation_position="top left")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="time (s)",
                      yaxis_title="cm off the ice", showlegend=False, title="Timeline")
    st.plotly_chart(fig, width="stretch")

with t_jumps:
    if not jumps:
        st.info("No jumps detected in this clip.")
    for j in jumps:
        head = f"Jump {j.n} · {j.code} · {j.jump_type} · {j.t_takeoff:.1f}s" + ("  ⚠️ fall" if j.fall else "")
        with st.expander(head, expanded=(len(jumps) <= 3)):
            a, b = st.columns([1, 2])
            with a:
                st.markdown(f"<div class='code'>{j.code}</div>", unsafe_allow_html=True)
                call_txt = {"clean": "✅ Fully rotated", "q": "🟡 Quarter short (q)",
                            "<": "🟠 Under-rotated (<)", "<<": "🔴 Downgraded (<<)"}[j.rotation_call]
                st.markdown(call_txt + (f" · BV ≈ {j.base_value:.2f}" if j.base_value is not None else ""))
                new_type = st.selectbox("Jump type (correct me if wrong)", C.JUMP_TYPES,
                                        index=C.JUMP_TYPES.index(j.jump_type), key=f"type_{j.n}")
                if new_type != j.jump_type:
                    regrade_jump(S, j, new_type)
                    st.session_state.report = build_report(S, jumps, spins, height_cm / 100)
                    st.rerun()
                st.caption(f"Type confidence {j.type_confidence:.0%} (else maybe {j.type_alternative}). "
                           + " ".join(r[0].upper() + r[1:] + "." for r in j.type_reasons))
            with b:
                k = st.columns(4)
                k[0].metric("Rotations", f"{j.rev_measured:.2f}", f"of {j.nominal_rev:g}", delta_color="off")
                k[1].metric("Air time", f"{j.air_time:.2f}s")
                k[2].metric("Height", f"{j.height_m * 100:.0f} cm")
                k[3].metric("Rotation speed", f"{j.rev_per_sec:.1f} rev/s")
                k = st.columns(4)
                k[0].metric("Approach", speed_txt(j.approach_speed))
                k[1].metric("At take-off", speed_txt(j.takeoff_speed))
                k[2].metric("Exit", speed_txt(j.exit_speed))
                k[3].metric("Distance", f"{j.distance_tl * tl_m:.1f} m")
                st.caption(f"Rotation-count confidence: {j.rev_confidence}")

            try:
                st.image(keyframe_strip(path, tr, j), width="stretch")
            except Exception as e:  # noqa: BLE001
                st.caption(f"(couldn't draw key frames: {e})")

            ci, cg = st.columns([3, 2])
            with ci:
                st.markdown("**Fix this**")
                if not j.issues:
                    st.markdown("Nothing flagged on this one 🎉")
                for iss in sorted(j.issues, key=lambda x: -x.severity):
                    st.markdown(f"<div class='issue sev{iss.severity}'><b>{iss.title}</b> "
                                f"<span class='small'>· {SEV[iss.severity]}</span><br>{iss.detail}<br>"
                                f"👉 {iss.tip}</div>", unsafe_allow_html=True)
            with cg:
                st.markdown("**Good**")
                for s in j.strengths:
                    st.markdown(f"<div class='good'>{s}</div>", unsafe_allow_html=True)

            clips = st.session_state.setdefault("clips", {})
            ck = (j.n, j.code)
            if ck in clips and os.path.exists(clips[ck]):
                st.video(clips[ck])
            elif st.button("🎬 Make slow-motion clip with skeleton", key=f"clip_{j.n}"):
                with st.spinner("Rendering clip…"):
                    out = os.path.join(tempfile.gettempdir(), f"skate_{vid_hash}_j{j.n}.mp4")
                    try:
                        clips[ck] = jump_clip(path, tr, S, j, out)
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Couldn't render clip: {e}")

with t_spins:
    if not spins:
        st.info("No spins detected (needs at least 2 revolutions on the spot).")
    for s in spins:
        with st.expander(f"Spin {s.n} · {s.position} · {s.t_start:.1f}s", expanded=True):
            k = st.columns(4)
            k[0].metric("Revolutions", f"{s.revolutions:.1f}")
            k[1].metric("Duration", f"{s.duration:.1f}s")
            k[2].metric("Speed", f"{s.rev_per_sec:.1f} rev/s")
            k[3].metric("Travel", f"{s.travel_tl * tl_m:.1f} m")
            for iss in s.issues:
                st.markdown(f"<div class='issue sev{iss.severity}'><b>{iss.title}</b><br>{iss.detail}<br>👉 {iss.tip}</div>",
                            unsafe_allow_html=True)
            for g in s.strengths:
                st.markdown(f"<div class='good'>{g}</div>", unsafe_allow_html=True)

with t_speed:
    kmh = np.nan_to_num(S.speed) * tl_m * 3.6
    fig = go.Figure(go.Scatter(x=S.t, y=kmh, line=dict(color="#4cc3d9", width=1.5), name="speed"))
    for j in jumps:
        fig.add_vline(x=j.t_takeoff, line_dash="dot", line_color="#8e6ff7",
                      annotation_text=j.code, annotation_position="top")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="time (s)",
                      yaxis_title="km/h", title="Skating speed over the ice")
    st.plotly_chart(fig, width="stretch")
    if jumps:
        df = pd.DataFrame([{"Jump": f"{j.n} {j.code}",
                            "Approach (km/h)": round(j.approach_speed * tl_m * 3.6, 1),
                            "Take-off (km/h)": round(j.takeoff_speed * tl_m * 3.6, 1),
                            "Exit (km/h)": round(j.exit_speed * tl_m * 3.6, 1),
                            "Kept speed (%)": round(100 * j.exit_speed / j.approach_speed) if j.approach_speed else None}
                           for j in jumps])
        st.dataframe(df, hide_index=True, width="stretch")
    st.caption("Speed is measured relative to the ice: camera panning is estimated from the background and removed. "
               "Values scale with the height you set in the sidebar.")

with t_export:
    js = to_jsonable(R)
    st.download_button("Download full report (JSON)", json.dumps(js, indent=2, default=float),
                       file_name="skate_report.json", mime="application/json")
    if jumps:
        buf = io.StringIO()
        pd.DataFrame([{k: v for k, v in jj.items() if k not in ("issues", "strengths", "type_reasons", "metrics")}
                      | {"issues": "; ".join(i["title"] for i in jj["issues"])} for jj in js["jumps"]]).to_csv(buf, index=False)
        st.download_button("Download jumps (CSV)", buf.getvalue(), file_name="skate_jumps.csv", mime="text/csv")

st.divider()
st.caption("Measurements come from a single camera and AI pose estimation, so treat them as a training aid, "
           "not an official judging call. Jump-type detection is a best guess — correct it in the Jumps tab "
           "and everything re-scores.")
