"""Pose extraction: runs MediaPipe PoseLandmarker over a video and also
estimates camera panning from background optical flow, so speed is measured
relative to the ice rather than to the screen."""
from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
FLOW_WIDTH = 192  # px width used for background flow (cheap)


@dataclass
class PoseTrack:
    fps: float                 # effective analysis fps (after stride)
    src_fps: float
    width: int                 # original video size
    height: int
    frame_idx: np.ndarray      # (T,) original frame numbers
    t: np.ndarray              # (T,) seconds
    img: np.ndarray            # (T,33,3) x,y in pixels of ORIGINAL frame, visibility
    world: np.ndarray          # (T,33,3) metres, hip-centred
    valid: np.ndarray          # (T,) bool – a pose was found
    pan: np.ndarray            # (T,2) background motion, original px per analysed frame
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.t)


def get_model_path(variant: str = "full") -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"pose_landmarker_{variant}.task"
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URLS[variant], tmp)
    os.replace(tmp, path)
    return path


def video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(fps) or fps <= 1 or fps > 1000:
        fps = 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": float(fps), "frames": n, "width": w, "height": h,
            "duration": n / fps if n else 0.0}


def _bg_flow(prev_small, small, bbox_small):
    """Median optical flow of textured background pixels (person masked out)."""
    flow = cv2.calcOpticalFlowFarneback(prev_small, small, None, 0.5, 2, 13, 2, 5, 1.1, 0)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    mask = mag > np.percentile(mag, 60)
    if bbox_small is not None:
        x0, y0, x1, y1 = bbox_small
        mask[max(0, y0):y1, max(0, x0):x1] = False
    if mask.sum() < 50:
        return np.zeros(2, np.float32)
    return np.array([np.median(flow[..., 0][mask]), np.median(flow[..., 1][mask])], np.float32)


def extract_poses(video_path: str, variant: str = "full", max_width: int = 640,
                  start_s: float = 0.0, end_s: float | None = None,
                  max_fps: float = 60.0, progress_cb=None) -> PoseTrack:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    info = video_info(video_path)
    src_fps = info["fps"]
    stride = max(1, int(round(src_fps / max_fps))) if max_fps else 1
    fps = src_fps / stride

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(get_model_path(variant))),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    W, H = info["width"], info["height"]
    start_f = int(start_s * src_fps)
    end_f = int(end_s * src_fps) if end_s else (info["frames"] or 10**9)
    if start_f > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    total = max(1, (end_f - start_f) // stride)

    frame_idx, img_l, world_l, valid_l, pan_l = [], [], [], [], []
    prev_small, last_bbox = None, None
    f = start_f
    t0 = time.time()
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while f < end_f:
            ok, frame = cap.read()
            if not ok:
                break
            if (f - start_f) % stride:
                f += 1
                continue
            H, W = frame.shape[:2]
            scale = min(1.0, max_width / W)
            small_rgb = cv2.cvtColor(
                cv2.resize(frame, (int(W * scale), int(H * scale))) if scale < 1 else frame,
                cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(small_rgb))
            res = landmarker.detect_for_video(mp_img, int(round(f * 1000.0 / src_fps)))

            pts = np.full((33, 3), np.nan, np.float32)
            wpts = np.full((33, 3), np.nan, np.float32)
            ok_pose = bool(res.pose_landmarks)
            if ok_pose:
                lm = res.pose_landmarks[0]
                pts = np.array([[p.x * W, p.y * H, p.visibility or 0.0] for p in lm], np.float32)
                if res.pose_world_landmarks:
                    wpts = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]], np.float32)
                xs, ys = pts[:, 0], pts[:, 1]
                pad_x, pad_y = 0.35 * (xs.max() - xs.min()), 0.2 * (ys.max() - ys.min())
                last_bbox = (xs.min() - pad_x, ys.min() - pad_y, xs.max() + pad_x, ys.max() + pad_y)

            # camera pan from background flow
            fs = FLOW_WIDTH / W
            gray = cv2.cvtColor(cv2.resize(frame, (FLOW_WIDTH, max(8, int(H * fs)))), cv2.COLOR_BGR2GRAY)
            if prev_small is None:
                pan = np.zeros(2, np.float32)
            else:
                bb = None if last_bbox is None else tuple(int(v * fs) for v in last_bbox)
                pan = _bg_flow(prev_small, gray, bb) / fs
            prev_small = gray

            frame_idx.append(f)
            img_l.append(pts)
            world_l.append(wpts)
            valid_l.append(ok_pose)
            pan_l.append(pan)
            f += 1
            if progress_cb and len(frame_idx) % 10 == 0:
                done = len(frame_idx) / total
                el = time.time() - t0
                progress_cb(min(done, 1.0), el / max(done, 1e-6) - el)
    cap.release()

    frame_idx = np.array(frame_idx, int)
    return PoseTrack(
        fps=fps, src_fps=src_fps, width=W, height=H,
        frame_idx=frame_idx, t=frame_idx / src_fps,
        img=np.array(img_l, np.float32).reshape(-1, 33, 3),
        world=np.array(world_l, np.float32).reshape(-1, 33, 3),
        valid=np.array(valid_l, bool),
        pan=np.array(pan_l, np.float32).reshape(-1, 2),
        meta={"variant": variant, "stride": stride, "start_s": start_s},
    )
