import random

from PIL import Image
import torch
import torchvision.transforms.functional as F

class RandomVerticalFlip(object):
  """Vertically flip the given image and its segmentation mask image randomly with a given probability.
  Args:
    p (float): probability that the image and its segmentation mask being flipped (default value: 0.5).
  """
  def __init__(self, p: float=0.5) -> None:
    self.p = p

  def __call__(self, img: torch.Tensor | Image.Image, lbl: torch.Tensor | Image.Image) -> tuple[torch.Tensor | Image.Image, torch.Tensor | Image.Image]:
    if random.random() < self.p:
      return F.vflip(img), F.vflip(lbl)
    return img, lbl

  def __repr__(self) -> str:
    return self.__class__.__name__ + "(p={})".format(self.p)
