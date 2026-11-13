"""Reference values used by the analyzer.

Everything here is an editable estimate — tweak the numbers to match your
coach's targets or the current ISU Scale of Values.
"""

# MediaPipe Pose landmark indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

SKELETON = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

JUMP_TYPES = ["Axel", "Toe Loop", "Salchow", "Loop", "Flip", "Lutz", "Waltz jump", "Unknown"]
JUMP_ABBR = {
    "Axel": "A", "Toe Loop": "T", "Salchow": "S", "Loop": "Lo",
    "Flip": "F", "Lutz": "Lz", "Waltz jump": "W", "Unknown": "?",
}

# ISU base values (2022-23 Scale of Values, Communication 2475). Check the
# latest ISU communication before relying on these for real scoring.
BASE_VALUES = {
    "1T": 0.40, "1S": 0.40, "1Lo": 0.50, "1F": 0.50, "1Lz": 0.60, "1A": 1.10,
    "2T": 1.30, "2S": 1.30, "2Lo": 1.70, "2F": 1.80, "2Lz": 2.10, "2A": 3.30,
    "3T": 4.20, "3S": 4.30, "3Lo": 4.90, "3F": 5.30, "3Lz": 5.90, "3A": 8.00,
    "4T": 9.50, "4S": 9.70, "4Lo": 10.50, "4F": 11.00, "4Lz": 11.50, "4A": 12.50,
}
UNDER_ROTATED_FACTOR = 0.80  # "<" call keeps 80% of base value

# Rough air-time targets (seconds) by number of rotations, for feedback.
AIR_TIME_TARGET = {0.5: 0.35, 1.0: 0.40, 1.5: 0.45, 2.0: 0.52, 2.5: 0.56, 3.0: 0.62, 3.5: 0.66, 4.0: 0.70, 4.5: 0.74}

# Rotation-shortfall thresholds (in revolutions) for ISU-style calls
Q_SHORT = 0.20        # >= this: quarter short ("q")
UR_SHORT = 0.30       # >= this: under-rotated ("<")
DOWNGRADE_SHORT = 0.50  # >= this: downgraded ("<<")

# Torso (shoulder-centre to hip-centre) as a fraction of standing height
TORSO_TO_HEIGHT = 0.30
G = 9.81
