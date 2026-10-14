import os

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.extra as T
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.extra import make_grid

def main():
  torch.manual_seed(0)

  transform = T.RandomHorizontalFlip(p=1.0)
  img = Image.open(
    os.path.join("assets", "examples", "VOC2010", "JPEGImages", "living_room.jpg")
  ).convert("RGB")
  mask = Image.open(
    os.path.join("assets", "examples", "VOC2010", "SegmentationObject", "living_room.png")
  ).convert("RGB")

  imgs = []
  masks = []
  for _ in range(4):
    img, mask = transform(img, mask)
    imgs.append(F.to_tensor(img))
    masks.append(mask)
  grid = make_grid(imgs, masks, nrow=4)
  plt.imshow(grid.permute(1, 2, 0))
  plt.axis("off")
  plt.show()

if __name__ == "__main__":
  main()