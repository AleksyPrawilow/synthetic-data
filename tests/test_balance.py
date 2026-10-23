import random
from collections import Counter
from PIL import Image

from synthdata.balance import augment, oversample


def test_oversample_length_and_membership():
    photos = [f"Egyptian_Mau_{i:03d}" for i in range(15)]
    budget = 55
    rng = random.Random(42)

    result = oversample(photos, budget, rng)
    assert len(result) == budget
    assert set(result).issubset(set(photos))


def test_oversample_distribution_is_even():
    photos = [f"Egyptian_Mau_{i:03d}" for i in range(15)]
    budget = 55
    rng = random.Random(42)

    result = oversample(photos, budget, rng)
    counts = Counter(result)
    assert len(counts) == 15
    assert max(counts.values()) - min(counts.values()) <= 1
    # 55 = 15 * 3 + 10, so 10 photos appear 4 times and 5 appear 3 times
    assert sorted(counts.values()) == [3] * 5 + [4] * 10


def test_oversample_reproducibility():
    photos = [f"Egyptian_Mau_{i:03d}" for i in range(15)]
    budget = 55

    res1 = oversample(photos, budget, random.Random(123))
    res2 = oversample(photos, budget, random.Random(123))
    res3 = oversample(photos, budget, random.Random(999))

    assert res1 == res2
    assert res1 != res3


def test_augment_returns_pil_image():
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    augmented = augment(img)

    assert isinstance(augmented, Image.Image)
    assert augmented.size == (224, 224)

