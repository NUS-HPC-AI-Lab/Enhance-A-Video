NUM_FRAMES = None
ENHANCE_WEIGHT = None
ENABLE_ENHANCE = False


def set_num_frames(num_frames: int):
    global NUM_FRAMES
    NUM_FRAMES = num_frames


def get_num_frames() -> int:
    return NUM_FRAMES


def enable_enhance():
    global ENABLE_ENHANCE
    ENABLE_ENHANCE = True


def is_enhance_enabled() -> bool:
    return ENABLE_ENHANCE


def set_enhance_weight(enhance_weight: float):
    global ENHANCE_WEIGHT
    ENHANCE_WEIGHT = enhance_weight


def get_enhance_weight() -> float:
    return ENHANCE_WEIGHT
