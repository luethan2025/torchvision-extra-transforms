from ._geometry.random_horizontal_flip import RandomHorizontalFlip as _RandomHorizontalFlip
from ._geometry.random_vertical_flip import RandomVerticalFlip as _RandomVerticalFlip

class RandomHorizontalFlip(_RandomHorizontalFlip):
  __module__ = "torchvision.transforms.extra._geometry"

class RandomVerticalFlip(_RandomVerticalFlip):
  __module__ = "torchvision.transforms.extra._geometry"
