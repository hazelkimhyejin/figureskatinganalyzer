"""Synthetic skater generator for testing the analysis engine without video."""
import numpy as np

from skate.pose import PoseTrack

# body-frame points (metres): x right, y up, z forward
BASE = {
    0: (0, 0.78, 0.10), 11: (-0.2, 0.55, 0), 12: (0.2, 0.55, 0),
    13: (-0.32, 0.30, 0), 14: (0.32, 0.30, 0), 15: (-0.38, 0.05, 0.1), 16: (0.38, 0.05, 0.1),
    23: (-0.12, 0.05, 0), 24: (0.12, 0.05, 0), 25: (-0.12, -0.42, 0.05), 26: (0.12, -0.42, 0.05),
    27: (-0.12, -0.88, 0), 28: (0.12, -0.88, 0), 29: (-0.12, -0.93, -0.06), 30: (0.12, -0.93, -0.06),
    31: (-0.12, -0.93, 0.16), 32: (0.12, -0.93, 0.16),
}


def body(tuck=0.0, knee=0.0, split=0.0, pick_left=None):
    pts = {k: np.array(v, float) for k, v in BASE.items()}
    for w, e in ((15, 13), (16, 14)):
        s = np.sign(pts[w][0])
        pts[w] = (1 - tuck) * pts[w] + tuck * np.array([0.06 * s, 0.38, 0.12])
        pts[e] = (1 - tuck) * pts[e] + tuck * np.array([0.18 * s, 0.35, 0.05])
    # knee bend: drop the hips/upper body, push knees forward
    drop = 0.25 * knee
    for k in (0, 11, 12, 13, 14, 15, 16, 23, 24):
        pts[k][1] -= drop
    for k in (25, 26):
        pts[k][1] -= drop * 0.6
        pts[k][2] += 0.28 * knee
    if split:
        # one leg reaching back (picking) – behind = -z
        side = (27, 29, 31, 25) if pick_left else (28, 30, 32, 26)
        for k in side:
            pts[k][2] -= 0.55 * split
    # fill face (1-10) from the nose and fingers (17-22) from the wrists, like MediaPipe would
    for k in range(1, 11):
        pts[k] = pts[0] + np.array([0.03 * ((k % 2) * 2 - 1), 0.02, 0])
    for k in (17, 19, 21):
        pts[k] = pts[15].copy()
    for k in (18, 20, 22):
        pts[k] = pts[16].copy()
    return pts


def render(frames, fps, scale=200.0, W=1280, H=720, noise=0.004, seed=0):
    """frames: list of dict(x, lift, yaw, tuck, knee, split, pick_left)."""
    rng = np.random.default_rng(seed)
    T = len(frames)
    img = np.zeros((T, 33, 3), np.float32)
    world = np.zeros((T, 33, 3), np.float32)
    ground = H * 0.8
    for i, f in enumerate(frames):
        pts = body(f.get("tuck", 0), f.get("knee", 0), f.get("split", 0), f.get("pick_left"))
        c, s = np.cos(f["yaw"]), np.sin(f["yaw"])
        for k, (x, y, z) in pts.items():
            xr, zr = x * c - z * s, x * s + z * c
            world[i, k] = (xr, -y, zr)
            img[i, k, 0] = f["x"] * scale + xr * scale + rng.normal(0, noise * scale)
            img[i, k, 1] = ground - (y + 0.93 + f["lift"]) * scale + rng.normal(0, noise * scale)
            img[i, k, 2] = 0.95
        world[i] += rng.normal(0, 0.01, world[i].shape)
    t = np.arange(T) / fps
    return PoseTrack(fps=fps, src_fps=fps, width=W, height=H, frame_idx=np.arange(T), t=t,
                     img=img, world=world, valid=np.ones(T, bool), pan=np.zeros((T, 2), np.float32))


FACE_RIGHT = -np.pi / 2   # forward (+z) points to image +x
FACE_LEFT = np.pi / 2


def program(fps=60, axel_rev=1.5, toe_rev=1.0, spin=True, airtime=0.5):
    """Forward approach → Axel; glide; backward toe-pick jump; optional upright spin."""
    F = []
    x = 0.5
    v = 3.0  # m/s

    def glide(sec, yaw, vel, **kw):
        nonlocal x
        for _ in range(int(sec * fps)):
            x += vel / fps
            F.append(dict(x=x, lift=0, yaw=yaw, **kw))

    def jump(yaw0, rev, vel, air):
        nonlocal x
        n = int(air * fps)
        for k in range(n):
            u = (k + 0.5) / n
            x += vel / fps
            lift = 9.81 * air ** 2 / 2 * u * (1 - u)
            F.append(dict(x=x, lift=lift, yaw=yaw0 - 2 * np.pi * rev * u, tuck=min(1, u * 4)))
        return yaw0 - 2 * np.pi * rev

    # forward approach, deep knee before take-off
    glide(1.5, FACE_RIGHT, v)
    glide(0.35, FACE_RIGHT, v, knee=0.9)
    yaw = jump(FACE_RIGHT, axel_rev, v, airtime)
    glide(0.3, yaw, v * 0.8, knee=0.6)
    glide(1.5, yaw, v * 0.8)
    # backward toe-pick entry (facing left while travelling right)
    yaw = FACE_LEFT
    glide(1.2, yaw, v)
    glide(0.3, yaw, v, knee=0.8, split=1.0, pick_left=True)
    yaw = jump(yaw, toe_rev, v, airtime * 0.9)
    glide(1.5, yaw, v * 0.8)
    if spin:
        for k in range(int(3.0 * fps)):
            F.append(dict(x=x, lift=0, yaw=yaw - 2 * np.pi * 2.5 * k / fps, tuck=0.6))
        glide(1.0, yaw, 1.0)
    return F
