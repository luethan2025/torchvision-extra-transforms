from ._geometry.random_horizontal_flip import RandomHorizontalFlip as _RandomHorizontalFlip
from ._geometry.random_vertical_flip import RandomVerticalFlip as _RandomVerticalFlip
from ._geometry.center_crop import CenterCrop as _CenterCrop
from ._geometry.normalize import Normalize as _Normalize

class RandomHorizontalFlip(_RandomHorizontalFlip):
  __module__ = "torchvision.transforms.extra._geometry"

class RandomVerticalFlip(_RandomVerticalFlip):
  __module__ = "torchvision.transforms.extra._geometry"

class CenterCrop(_CenterCrop):
  __module__ = "torchvision.transforms.extra._geometry"

class Normalize(_Normalize):
  __module__ = "torchvision.transforms.extra._geometry"
