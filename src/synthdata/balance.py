"""Meeting 1: the two classical ways to add Egyptian Mau training examples.

Both functions are yours to write; `uv run synth-train --method oversample` or
`--method augment` uses them.
"""

import os
import random

from PIL import Image, ImageEnhance, ImageOps

AUGMENT_LEVEL = os.environ.get("AUGMENT_LEVEL", "moderate").lower()

try:
    from torchvision import transforms

    if AUGMENT_LEVEL == "strong":
        _TORCHVISION_TRANSFORM = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.08, 0.08),
                    scale=(0.88, 1.12),
                    shear=(-8, 8),
                ),
                transforms.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.25,
                    hue=0.04,
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
            ]
        )
    else:  # moderate (default)
        _TORCHVISION_TRANSFORM = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.04, 0.04),
                    scale=(0.95, 1.05),
                ),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15,
                ),
            ]
        )
except ImportError:
    _TORCHVISION_TRANSFORM = None


def oversample(
    minority_photos: list[str], budget: int, rng: random.Random
) -> list[str]:
    """Choose which Egyptian Mau training photos to repeat.

    Return a list of exactly `budget` names taken from `minority_photos`; names
    may repeat. Repeat the photos as evenly as possible and use `rng` for every
    random choice, so a run can be reproduced from its seed.
    """
    if not minority_photos:
        raise ValueError("minority_photos cannot be empty")
    if budget < 0:
        raise ValueError("budget cannot be negative")

    order = list(minority_photos)
    chosen: list[str] = []
    while len(chosen) < budget:
        rng.shuffle(order)
        chosen.extend(order)
    return chosen[:budget]


def augment(image: Image.Image) -> Image.Image:
    """Return a randomly transformed copy of one Egyptian Mau training photo.

    The training script calls this every time an added copy is loaded, so each
    epoch sees a new variant. Typical choices are crops, flips, rotations, and
    colour changes from `torchvision.transforms`. Keep the cat recognisable.
    """
    copy_img = image.copy()
    if _TORCHVISION_TRANSFORM is not None:
        return _TORCHVISION_TRANSFORM(copy_img)

    # Pure PIL fallback when running in minimal environments without torchvision
    if random.random() < 0.5:
        copy_img = ImageOps.mirror(copy_img)
    angle = random.uniform(-15.0, 15.0)
    copy_img = copy_img.rotate(angle, resample=Image.Resampling.BILINEAR)
    copy_img = ImageEnhance.Color(copy_img).enhance(random.uniform(0.85, 1.15))
    copy_img = ImageEnhance.Contrast(copy_img).enhance(random.uniform(0.85, 1.15))
    copy_img = ImageEnhance.Brightness(copy_img).enhance(random.uniform(0.85, 1.15))
    return copy_img
