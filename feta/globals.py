NUM_FRAMES = None
FETA_WEIGHT = None


def set_num_frames(num_frames: int):
    global NUM_FRAMES
    NUM_FRAMES = num_frames


def get_num_frames() -> int:
    return NUM_FRAMES


def set_feta_weight(feta_weight: float):
    global FETA_WEIGHT
    FETA_WEIGHT = feta_weight


def get_feta_weight() -> float:
    return FETA_WEIGHT
