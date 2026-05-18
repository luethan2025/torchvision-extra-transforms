import numbers
from typing import Sequence

from PIL import Image
import torch
import torchvision.transforms.functional as F

class CenterCrop(object):
  """Crops the given PIL Image at the center.
  Args:
    size (sequence or int): Desired output size of the crop. If size is an
      int instead of sequence like (h, w), a square crop (size, size) is
      made.
  """
  def __init__(self, size: float | Sequence[int]) -> None:
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size

  def __call__(self, img: Image.Image, lbl: Image.Image) -> tuple[Image.Image, Image.Image]:
    return F.center_crop(img, self.size), F.center_crop(lbl, self.size)

  def __repr__(self) -> str:
    return self.__class__.__name__ + "(size={0})".format(self.size)
