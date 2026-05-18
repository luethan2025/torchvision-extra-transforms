import os
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def voc_cmap(N: int=256, normalized: bool=False) -> np.ndarray:
  def bitget(byteval: int, idx: int) -> bool:
    return ((byteval & (1 << idx)) != 0)

  dtype = "float32" if normalized else "uint8"
  cmap = np.zeros((N, 3), dtype=dtype)
  for i in range(N):
    r = g = b = 0
    c = i
    for j in range(8):
      r = r | (bitget(c, 0) << 7 - j)
      g = g | (bitget(c, 1) << 7 - j)
      b = b | (bitget(c, 2) << 7 - j)
      c = c >> 3

    cmap[i] = np.array([r, g, b])

  cmap = cmap / 255 if normalized else cmap
  return cmap

class _VOCSegmentation(Dataset):
  """Pascal VOC Segmentation Dataset.
  Args:
    root (str): root directory of the VOC Dataset.
    year (str): dataset year.
    image_set (str, optional): 'train', 'trainval', or 'val'.
    transform (callable, optional): transformations from the `torchvision.transform.extra` module.
  """
  cmap = voc_cmap()
  def __init__(self, root: str, year: str="2012", image_set: str="train", transform: Optional[Callable]=None) -> None:
    super().__init__()
    valid_years = ["2012"]
    if year not in valid_years:
      raise ValueError()
    self.year = year
    valid_image_sets = ["train", "trainval", "val"]
    if image_set not in valid_image_sets:
      raise ValueError()
    self.image_set = image_set
    self.transform = transform

    self.root = os.path.expanduser(root)
    voc_root = os.path.join(self.root, "VOCdevkit", f"VOC{self.year}")

    image_dir = os.path.join(voc_root, "JPEGImages")
    mask_dir = os.path.join(voc_root, "SegmentationClass")
    splits_dir = os.path.join(voc_root, "ImageSets/Segmentation")

    split_f = os.path.join(splits_dir, image_set.rstrip("\n") + '.txt')
    if not os.path.exists(split_f):
      raise ValueError()
   
    with open(os.path.join(split_f), "r") as f:
      file_names = [x.strip() for x in f.readlines()]
    self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
    self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
    assert (len(self.images) == len(self.masks))

  def __getitem__(self, idx: int) -> tuple[Image.Image, Image.Image]:
    img = Image.open(self.images[idx]).convert("RGB")
    target = Image.open(self.masks[idx])
    if self.transform is not None:
      img, target = self.transform(img, target)
    return img, target
  
  def __len__(self) -> int:
    length = len(self.images)
    return length

  def __repr__(self) -> str:
    return self.__class__.__name__ + "(year={}, image_set={})".format(self.year, self.image_set)

  @classmethod
  def decode_target(cls, lbl: int) -> np.ndarray:
    return cls.cmap[lbl]

  @classmethod
  def decode_segmentation_image(cls, mask: np.ndarray) -> np.ndarray:
    img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for lbl, c in enumerate(cls.cmap):
      img[mask == lbl] = c
    return img
