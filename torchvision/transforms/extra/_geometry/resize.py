from typing import Sequence

from PIL import Image
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

class Resize(object):
  """Crops the given PIL Image at the center.
  Args:
    size (sequence or int): Desired output size. If size is a sequence like
        (h, w), output size will be matched to this. If size is an int,
        smaller edge of the image will be matched to this number.
        i.e, if height > width, then image will be rescaled to
        (size * height / width, size).
    interpolation (int, optional): Desired interpolation. Default is
        ``torchvision.transforms.InterpolationMode.BILINEAR``.
  """
  def __init__(self, size: int | Sequence[int], interpolation: torchvision.transforms.InterpolationMode=torchvision.transforms.InterpolationMode.BILINEAR) -> None:
    self.size = size
    self.interpolation = interpolation

  def __call__(self, img: Image.Image, lbl: Image.Image) -> tuple[Image.Image, Image.Image]:
    return F.resize(img, self.size, self.interpolation), F.resize(lbl, self.size, InterpolationMode.NEAREST)

  def __repr__(self) -> str:
    return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, self.interpolation.name) 
