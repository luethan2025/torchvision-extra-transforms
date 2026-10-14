import os

import PIL
from PIL import Image

def load_pascal_voc2010_segmentation_example() -> tuple[PIL.Image.Image, PIL.Image.Image]:
  img = Image.open(
    os.path.join("assets", "examples", "VOC2010", "JPEGImages", "living_room.jpg")
  ).convert("RGB")
  mask = Image.open(
    os.path.join("assets", "examples", "VOC2010", "SegmentationObject", "living_room.png")
  ).convert("RGB")
  return img, mask
