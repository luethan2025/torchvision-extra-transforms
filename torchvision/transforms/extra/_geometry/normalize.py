import numbers
from typing import Sequence

from PIL import Image
import torch
import torchvision.transforms.functional as F

class Normalize(object):
  """Normalize a tensor image with mean and standard deviation.
  Args:
    mean: (sequence or float): mean
    std: (sequence or float): standard deviation
  """
  def __init__(self, mean: float | Sequence[float], std: float | Sequence[float]) -> None:
    self.mean = mean
    self.std = std

  def __call__(self, img: torch.Tensor | Image.Image, lbl: torch.Tensor | Image.Image) -> tuple[torch.Tensor | Image.Image, torch.Tensor | Image.Image]:
    return F.normalize(img, self.mean, self.std), lbl

  def __repr__(self):
    return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
