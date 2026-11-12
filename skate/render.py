"""Drawing: skeleton overlays, key-frame strips and annotated slow-motion clips."""
from __future__ import annotations

import cv2
import numpy as np

from . import constants as C
from .analysis import Jump, Signals
from .pose import PoseTrack

LILAC = (255, 140, 190)     # BGR
ICE = (255, 220, 120)
WARN = (60, 170, 255)
WHITE = (255, 255, 255)


def read_frame(path: str, frame_no: int):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def draw_pose(frame, pts, color=LILAC, thick=None):
    if pts is None or not np.isfinite(pts[:, :2]).all():
        return frame
    h = frame.shape[0]
    thick = thick or max(2, h // 240)
    for a, b in C.SKELETON:
        pa, pb = pts[a, :2], pts[b, :2]
        cv2.line(frame, tuple(int(v) for v in pa), tuple(int(v) for v in pb), color, thick, cv2.LINE_AA)
    for k in range(11, 33):
        cv2.circle(frame, tuple(int(v) for v in pts[k, :2]), thick + 1, WHITE, -1, cv2.LINE_AA)
    return frame


def _label(frame, text, org, scale=None, color=WHITE):
    h = frame.shape[0]
    scale = scale or max(0.5, h / 900)
    th = max(1, int(scale * 2))
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = org
    cv2.rectangle(frame, (x - 6, y - tht - 8), (x + tw + 6, y + 8), (40, 20, 40), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def _crop_to_skater(frame, pts, aspect=0.75, pad=0.45):
    H, W = frame.shape[:2]
    if pts is None or not np.isfinite(pts[:, :2]).all():
        return frame
    body = pts[[0] + list(range(11, 33)), :2]
    x0, y0 = body[:, 0].min(), body[:, 1].min()
    x1, y1 = body[:, 0].max(), body[:, 1].max()
    ch = (y1 - y0) * (1 + 2 * pad)
    ch = max(ch, H * 0.35)
    cw = ch * aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    xa, ya = int(max(0, cx - cw / 2)), int(max(0, cy - ch / 2))
    xb, yb = int(min(W, cx + cw / 2)), int(min(H, cy + ch / 2))
    return frame[ya:yb, xa:xb]


def keyframe_strip(path: str, tr: PoseTrack, j: Jump, height: int = 360) -> np.ndarray:
    """Take-off / mid-air / landing / check-out frames side by side (RGB)."""
    co_i = min(len(tr) - 1, j.landing_i + int(0.4 * tr.fps))
    picks = [(j.takeoff_i, "Take-off"), (j.peak_i, "Peak"), (j.landing_i, "Landing"), (co_i, "Check-out")]
    tiles = []
    for i, name in picks:
        frame = read_frame(path, tr.frame_idx[i])
        if frame is None:
            continue
        pts = tr.img[i]
        draw_pose(frame, pts)
        tile = _crop_to_skater(frame, pts)
        if tile.size == 0:
            tile = frame
        s = height / tile.shape[0]
        tile = cv2.resize(tile, (max(1, int(tile.shape[1] * s)), height))
        _label(tile, name, (12, 32), scale=0.7)
        tiles.append(tile)
    if not tiles:
        return np.zeros((height, height, 3), np.uint8)
    gap = np.full((height, 6, 3), 255, np.uint8)
    out = tiles[0]
    for t in tiles[1:]:
        out = np.hstack([out, gap, t])
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def jump_clip(path: str, tr: PoseTrack, S: Signals, j: Jump, out_path: str,
              slow: float = 0.4, before: float = 1.2, after: float = 1.0, max_w: int = 720) -> str:
    """Annotated slow-motion MP4 (H.264, plays in browsers) around one jump."""
    import imageio_ffmpeg

    f0 = int(tr.frame_idx[max(0, j.takeoff_i)] - before * tr.src_fps)
    f1 = int(tr.frame_idx[min(len(tr) - 1, j.landing_i)] + after * tr.src_fps)
    f0 = max(0, f0)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    W, H = tr.width, tr.height
    s = min(1.0, max_w / W)
    ow, oh = int(W * s) // 2 * 2, int(H * s) // 2 * 2
    out_fps = max(5.0, tr.src_fps * slow)
    writer = imageio_ffmpeg.write_frames(out_path, (ow, oh), fps=out_fps, codec="libx264",
                                         pix_fmt_out="yuv420p", quality=7, macro_block_size=2,
                                         output_params=["-preset", "veryfast", "-movflags", "+faststart"])
    writer.send(None)
    to_t, ld_t = tr.t[j.takeoff_i], tr.t[j.landing_i]
    for f in range(f0, f1 + 1):
        ok, frame = cap.read()
        if not ok:
            break
        i = int(np.clip(np.searchsorted(tr.frame_idx, f), 0, len(tr) - 1))
        t = f / tr.src_fps
        if t < to_t:
            phase, col = "APPROACH", ICE
        elif t <= ld_t:
            phase, col = "IN THE AIR", LILAC
        else:
            phase, col = "LANDING", (140, 230, 140)
        draw_pose(frame, tr.img[i], col)
        frame = cv2.resize(frame, (ow, oh))
        sc = max(0.55, oh / 800)
        _label(frame, f"Jump {j.n}: {j.code}  ({j.jump_type})", (14, int(40 * sc)), sc)
        _label(frame, phase, (14, int(85 * sc)), sc, color=col)
        if t >= to_t:
            k = min(i, j.landing_i)
            rev_now = abs(S.yaw2[k] - S.yaw2[j.takeoff_i]) / (4 * np.pi)
            air_now = min(t, ld_t) - to_t
            _label(frame, f"rev {rev_now:.2f}   air {air_now:.2f}s", (14, int(130 * sc)), sc)
        _label(frame, f"{slow:g}x speed", (ow - int(150 * sc), oh - int(20 * sc)), sc * 0.8)
        writer.send(np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    writer.close()
    cap.release()
    return out_path
