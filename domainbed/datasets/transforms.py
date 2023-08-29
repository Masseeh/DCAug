from torchvision import transforms as T
from typing import List, Optional
import torchvision.transforms.functional as F
from .datasets import IMG_SIZE

import torch, math
from torch import Tensor
import numpy as np

def tf_discrete_range(r, num_bins=31, include_id=True):
    space = {
        "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
        "AutoContrast": (torch.tensor(0.0), False),
        "Equalize": (torch.tensor(0.0), False),
        "Gray": (torch.tensor(0.0), False)
    }
    if include_id:
        space["Identity"] = (torch.tensor(0.0), False)

    if r == 'default':
        space['ShearX'] = (torch.linspace(0.0, 0.3, num_bins), True)
        space['ShearY'] = (torch.linspace(0.0, 0.3, num_bins), True)
        space['TranslateX'] = (torch.linspace(0.0, 0.45 * IMG_SIZE, num_bins), True)
        space['TranslateY'] = (torch.linspace(0.0, 0.45 * IMG_SIZE , num_bins), True)
        space['Rotate'] = (torch.linspace(0.0, 30.0, num_bins), True)
        space['Color'] = (torch.linspace(0.0, 0.9, num_bins), True)
        space['Contrast'] = (torch.linspace(0.0, 0.9, num_bins), True)
        space['Brightness'] = (torch.linspace(0.0, 0.9, num_bins), True)
        space['Sharpness'] = (torch.linspace(0.0, 0.9, num_bins), True)
        space['Posterize'] = (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False)
    
    elif r == 'wide':
        space['ShearX'] = (torch.linspace(0.0, 0.99, num_bins), True)
        space['ShearY'] = (torch.linspace(0.0, 0.99, num_bins), True)
        space['TranslateX'] = (torch.linspace(0.0, 32, num_bins), True)
        space['TranslateY'] = (torch.linspace(0.0, 32, num_bins), True)
        space['Rotate'] = (torch.linspace(0.0, 135.0, num_bins), True)
        space['Color'] = (torch.linspace(0.0, 0.99, num_bins), True)
        space['Contrast'] = (torch.linspace(0.0, 0.99, num_bins), True)
        space['Brightness'] = (torch.linspace(0.0, 0.99, num_bins), True)
        space['Sharpness'] = (torch.linspace(0.0, 0.99, num_bins), True)
        space['Posterize'] = (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False)

    return space

def tf_continuous_range(r, include_id=False):
    space = {
            "ShearX": (-0.3, 0.3),
            "ShearY": (-0.3, 0.3),
            "TranslateX": (-0.45 * IMG_SIZE, 0.45 * IMG_SIZE),
            "TranslateY": (-0.45 * IMG_SIZE, 0.45 * IMG_SIZE),
            "Rotate": (-30.0, 30.0),
            "Contrast": (-0.9, 0.9),
            "Brightness": (-0.9, 0.9),
            "Color": (-0.9, 0.9),
            "Sharpness": (-0.9, 0.9),
            "Posterize": (0, 0.5),
            "Solarize": (255, 0),
            "AutoContrast": (0,),
            "Equalize": (0,),
            "Gray": (0,),
        }
    if include_id:
        space["Identity"] = (0,)

    if r == 'default':
        return space
    
    elif r == 'wide':
        space['ShearX'] = (-0.99, 0.99)
        space['ShearY'] = (-0.99, 0.99)
        space['Rotate'] = (-135.0, 135.0)
        space['TranslateX'] = (-32.0, 32.0)
        space['TranslateY'] = (-32.0, 32.0)
        space['Color'] = (-0.99, 0.99)
        space['Contrast'] = (-0.99, 0.99)
        space['Brightness'] = (-0.99, 0.99)
        space['Sharpness'] = (-0.99, 0.99)
        space['Posterize'] = (0.0, 0.75)

    elif r == 'wider':
        space['ShearX'] = (-1.0, 1.0)
        space['ShearY'] = (-1.0, 1.0)
        space['Rotate'] = (-135.0, 135.0)
        space['TranslateX'] = (-1.0 * IMG_SIZE , 1.0 * IMG_SIZE)
        space['TranslateY'] = (-1.0 * IMG_SIZE, 1.0 * IMG_SIZE)
        space['Color'] = (-10.0, 10.0)
        space['Contrast'] = (-10.0, 10.0)
        space['Brightness'] = (-1.0, 10.0)
        space['Sharpness'] = (-10.0, 10.0)
        space['Posterize'] = (0.0, 8.0)
    
    return space

def apply_op(img: Tensor, op_name: str, magnitude: float,
              interpolation: F.InterpolationMode = T.InterpolationMode.BILINEAR, fill: Optional[List[float]] = None):
    if op_name == "ShearX":
        img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[math.degrees(magnitude), 0.0],
                       interpolation=interpolation, fill=fill)
    elif op_name == "ShearY":
        img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[0.0, math.degrees(magnitude)],
                       interpolation=interpolation, fill=fill)
    elif op_name == "TranslateX":
        img = F.affine(img, angle=0.0, translate=[int(magnitude), 0], scale=1.0,
                       interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "TranslateY":
        img = F.affine(img, angle=0.0, translate=[0, int(magnitude)], scale=1.0,
                       interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Gray":
        img = F.rgb_to_grayscale(img, num_output_channels=3)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError("The provided operator {} is not recognized.".format(op_name))
    return img

class MyRandAugment(torch.nn.Module):
    def __init__(self, num_ops: int = 1, num_samples: int = 1, magnitude: int = 9, rng: str = 'wide', strategy: str = 'ta',
                 interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
                 fill: Optional[List[float]] = None) -> None:

        super().__init__()
        self.num_ops = num_ops
        self.num_samples = num_samples
        self.magnitude = magnitude
        self.interpolation = interpolation
        self.fill = fill
        self.strategy = strategy.lower()
        self.space = tf_discrete_range(rng, 31) if self.strategy in ['ta', 'ra'] else tf_continuous_range(rng) 

        if self.strategy == 'autoaugment':
            self.autoaug = T.AutoAugment(interpolation=T.InterpolationMode.BILINEAR)

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """

        if self.strategy == 'autoaugment':
            return [self.autoaug(img)]

        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self.space

        rounds = []
        r0 = np.random.choice(len(op_meta), size=self.num_ops)
        rounds.append(r0)
        for _ in range(self.num_samples - 1):
            r1 = np.random.choice(len(op_meta), size=self.num_ops)
            while not np.all(r1 != r0):
                r1 = np.random.choice(len(op_meta), size=self.num_ops)
            rounds.append(r1)

        out_images = []

        for r in rounds:

            tf_img = img.copy()
            for op_idx in r:

                op_name = list(op_meta.keys())[op_idx]

                if self.strategy in ['ta', 'ra']:
                    magnitudes, signed = op_meta[op_name]

                    if self.magnitude == -1: # TA
                        magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item()) \
                        if magnitudes.ndim > 0 else 0.0
                    else: # RA
                        magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0

                    if signed and torch.randint(2, (1,)):
                        magnitude *= -1.0

                else:
                    magnitudes = op_meta[op_name]
                    magnitude = (torch.rand(1) * (magnitudes[1] - magnitudes[0]) + magnitudes[0]).item() if len(magnitudes) > 1 else 0.0 
                
                tf_img = apply_op(tf_img, op_name, magnitude, interpolation=self.interpolation, fill=fill)
        
            out_images.append(tf_img)

        return out_images

_basic = [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

_aug = [
        T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        T.RandomGrayscale(p=0.1),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

basic = T.Compose(_basic)
aug = T.Compose(_aug)