import numbers
from typing import Sequence

from PIL import Image
import torchvision.transforms.functional as F

class GaussianBlur(object):
  """Applies a Gaussian blur to a given PIL Image.
  Args:
    kernel_size (sequence): Size of the Gaussian filter.
    sigma (sequence): Standard deviation of the Gaussian.
  """
  def __init__(self, kernel_size: numbers.Number | Sequence[numbers.Number], sigma: Sequence[numbers.Number]) -> None:
    self.kernel_size = kernel_size
    self.sigma = sigma

  def __call__(self, img: Image.Image, lbl: Image.Image) -> tuple[Image.Image, Image.Image]:
    return F.gaussian_blur(img, self.kernel_size, self.sigma), lbl

  def __repr__(self) -> str:
    return self.__class__.__name__ + "(kernel_size={0}, sigma={1})".format(self.kernel_size, self.sigma)
