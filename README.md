# ⛸️ Skate Analyzer

Upload a figure-skating video of any length and get a jump-by-jump breakdown:
what jump it was, how many rotations you actually completed, air time, height,
speed into and out of the jump, and the specific technique fixes to work on.

## What it does

| | |
|---|---|
| **Detects** | Axel, Toe Loop, Salchow, Loop, Flip, Lutz, Waltz jump, combinations, spins (upright / sit / camel) |
| **Measures** | Rotations, air time, jump height, rotation speed (rev/s), approach / take-off / exit speed (km/h), distance, knee bend, arm-tuck timing, air position, axis tilt |
| **Calls** | ISU-style rotation calls — clean, `q`, `<`, `<<` — falls / hand-downs, and an estimated base value |
| **Coaches** | Ranked "what to work on" list with tips, "what's working", per-jump issues, slow-motion clips with skeleton overlay and live rotation counter |
| **Exports** | Full JSON report and a CSV of all jumps |

If the app names a jump wrong, pick the right one in the **Jumps** tab — the
rotation call, base value and feedback re-score instantly.

## How it works

1. **Pose tracking** — Google MediaPipe PoseLandmarker finds 33 body points in every frame (2D image points + 3D body-centred points).
2. **Camera compensation** — background optical flow estimates how the camera pans, so speed is measured against the ice, not the screen.
3. **Jump detection** — the lowest foot leaving the ice line marks take-off and landing; air time gives height (h = g·t²/8).
4. **Rotation counting** — body yaw from the 3D shoulder/hip line is unwrapped through the flight (using the doubled angle, so left/right label swaps when your back faces the camera don't break the count), cross-checked against shoulder-width oscillation.
5. **Jump type** — forward take-off → Axel; toe pick (both feet down, wide apart) → Toe Loop vs Flip/Lutz by which toe picks; counter-rotation before take-off → Lutz; edge jumps split into Salchow (free-leg sweep) vs Loop (legs crossed).
6. **Feedback rules** — each measurement is compared with coaching targets (see `skate/constants.py` and `_jump_issues` in `skate/analysis.py`); tweak them to match your coach.

## Getting the best results

- Film **side-on** with the whole body in frame, camera following smoothly.
- Use **phone slow-mo (120/240 fps)** — fast rotations at 30 fps are only sampled a few times per turn.
- One skater in the shot. Set your **height** in the sidebar for accurate km/h.
- Long videos work; use the time slider to analyse just the part you care about.

> Measurements come from one camera and AI pose estimation — use them as a
> training aid, not an official judging call.

---

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The pose model (~9 MB) downloads automatically into `models/` on first run.

Run the tests (uses a synthetic skater, no video needed):

```bash
pip install pytest
pytest -q
```

## Deploy: GitHub + Streamlit Community Cloud

### 1. Push to GitHub

Create an empty repository on github.com (e.g. `skate-analyzer`), then in this folder:

```bash
git init
git add .
git commit -m "Skate Analyzer"
git branch -M main
git remote add origin https://github.com/<your-username>/skate-analyzer.git
git push -u origin main
```

### 2. Publish on Streamlit

1. Go to **share.streamlit.io** and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub**.
3. Repository: `<your-username>/skate-analyzer`, branch `main`, main file `app.py`.
4. **Advanced settings → Python version: 3.12**.
5. Deploy. First build takes a few minutes (MediaPipe is large).

`packages.txt` installs the system libraries MediaPipe/OpenCV need, and
`.streamlit/config.toml` raises the upload limit to 1 GB.

### Notes for the cloud

- Free Streamlit Cloud has limited CPU and memory. A 1-minute 60 fps clip takes
  roughly 1–3 minutes to analyse with the `full` model. For long competition
  videos, trim with the time slider or switch to the `lite` model.
- Very large uploads (>500 MB) can hit the free tier's memory limit — compress
  or trim the video first.

## Project layout

```
app.py                 Streamlit UI
skate/pose.py          MediaPipe pose tracking + camera-pan estimation
skate/analysis.py      signals, jump/spin detection, classification, feedback
skate/render.py        skeleton overlay, key-frame strips, slow-mo clips
skate/constants.py     landmark indices, ISU base values, thresholds (editable)
tests/                 synthetic skater generator + unit tests
```
