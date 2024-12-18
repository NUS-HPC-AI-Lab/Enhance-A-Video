NUM_FRAMES = None
FETA_RATIO = None


def set_num_frames(num_frames: int):
    global NUM_FRAMES
    NUM_FRAMES = num_frames


def get_num_frames() -> int:
    return NUM_FRAMES


def set_feta_ratio(feta_ratio: float):
    global FETA_RATIO
    FETA_RATIO = feta_ratio


def get_feta_ratio() -> float:
    return FETA_RATIO
