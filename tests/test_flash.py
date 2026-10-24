import numpy as np

from evperc.flash import FlashDetector, light_feature


def _score(signal_fn, fps=15.0, seconds=2.5, seed=0):
    rng = np.random.default_rng(seed)
    det = FlashDetector()
    t = np.arange(0, seconds, 1 / fps)
    x = signal_fn(t) + rng.normal(0, 0.03, len(t))
    for ti, xi in zip(t, np.clip(x, 0, 1), strict=True):
        det.add(1, float(ti), float(xi))
    return det.score(1)


def test_blinking_light_scores_high_and_steady_or_slow_low():
    blink = _score(lambda t: 0.05 + 0.3 * (np.sin(2 * np.pi * 3.0 * t) > 0))
    steady = _score(lambda t: np.full_like(t, 0.05))
    slow = _score(lambda t: 0.05 + 0.1 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t)))
    assert blink > 0.8
    assert steady < 0.3 and slow < 0.3


def test_score_is_none_until_window_is_covered():
    det = FlashDetector()
    for i in range(5):
        det.add(7, i / 15.0, 0.1)
    assert det.score(7) is None


def test_light_feature_detects_red_and_blue_pixels_in_roof_region():
    crop = np.zeros((100, 60, 3), dtype=np.uint8)
    assert light_feature(crop) == 0.0
    crop[:20, :30] = (20, 20, 250)  # BGR red patch on the roof
    assert light_feature(crop) > 0.2
    crop2 = np.zeros((100, 60, 3), dtype=np.uint8)
    crop2[80:, :] = (20, 20, 250)  # red in the lower part (tail lights) is ignored
    assert light_feature(crop2) == 0.0
