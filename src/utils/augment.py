import albumentations as A


class DarkAug(object):
    """Extreme dark augmentation aiming at Aachen Day-Night."""
    def __init__(self) -> None:
        self.augmentor = A.Compose(
            [
                A.RandomBrightnessContrast(p=0.75,
                                           brightness_limit=(-0.6, 0.0),
                                           contrast_limit=(-0.5, 0.3)),
                A.Blur(p=0.1, blur_limit=(3, 9)),
                A.MotionBlur(p=0.2, blur_limit=(3, 25)),
                A.RandomGamma(p=0.1, gamma_limit=(15, 65)),
                A.HueSaturationValue(p=0.1, val_shift_limit=(-100, -40)),
            ],
            p=0.75,
        )

    def __call__(self, x):
        return self.augmentor(image=x)['image']


class MobileAug(object):
    """Random augmentations aiming at images of mobile/handhold devices."""
    def __init__(self):
        self.augmentor = A.Compose(
            [
                A.MotionBlur(p=0.25),
                A.ColorJitter(p=0.5),
                A.RandomRain(p=0.1),  # random occlusion
                A.RandomSunFlare(p=0.1),
                A.JpegCompression(p=0.25),
                A.ISONoise(p=0.25),
            ],
            p=1.0,
        )

    def __call__(self, x):
        return self.augmentor(image=x)['image']
    

class LowLightAug(object):
    """Extreme dark augmentation aiming at Aachen Day-Night."""
    def __init__(self) -> None:
        self.augmentor = A.Compose(
            [
                A.RandomBrightnessContrast(p=0.6,
                                           brightness_limit=(-0.2, 0.0),
                                           contrast_limit=(-0.5, 0.2)),
                A.Blur(p=0.1, blur_limit=(3, 9)),
                A.MotionBlur(p=0.2, blur_limit=(3, 25)),
                A.RandomGamma(p=0.1, gamma_limit=(30, 90)),
                A.HueSaturationValue(p=0.1, val_shift_limit=(-100, -40)),
                A.ToGray(p=0.05),
                A.ShotNoise(p=0.2),
                A.ISONoise(p=0.3),
                A.GaussNoise(std_range=(0.02, 0.07), p=0.1),
                A.RandomToneCurve(scale=0.3, p=0.1),
                A.OneOf(
                    [
                        A.RandomFog(fog_coef_range=(0.1, 0.25), p=0.7),
                        A.RandomRain(p=0.5),
                    ],
                    p=0.2,
                ),
                A.Spatter(cutout_threshold=(0.7, 0.85), std=(0.3, 0.4), intensity=(0.15, 0.3), p=0.1),
            ],
            p=1.0,
        )

    def __call__(self, x):
        return self.augmentor(image=x)['image']


def build_augmentor(method=None, **kwargs):
    if method == 'dark':
        return DarkAug()
    elif method == 'mobile':
        return MobileAug()
    elif method == 'low_light':
        return LowLightAug()
    elif method is None:
        return None
    else:
        raise ValueError(f'Invalid augmentation method: {method}')


if __name__ == '__main__':
    augmentor = build_augmentor('FDA')
