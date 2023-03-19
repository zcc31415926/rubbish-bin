import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class ImgProcessor:
    @classmethod
    def loadImage(cls, img_files, img_size=(256, 256), mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225), device='cuda'):
        t = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        imgs = []
        for file in img_files:
            img = Image.open(file).convert('RGB')
            imgs.append(t(img).unsqueeze(0))
        return torch.cat(imgs, dim=0).float().to(device)

    @classmethod
    def saveImage(cls, imgs, output_file, img_size=(256, 256)):
        if imgs.size(0) > 1:
            if not os.path.exists(output_file):
                os.makedirs(output_file)
        else:
            if not output_file.endswith('.png') and not output_file.endswith('.jpg'):
                output_file += '.png'
        for i in range(imgs.size(0)):
            img = imgs[i].detach()
            std, mean = torch.std_mean(img, dim=1, keepdim=True)
            img = (img - mean) / std * 0.5 + 0.5
            img = torch.clamp(img, 0, 1).permute(1, 2, 0).cpu().numpy() * 255
            img = Image.fromarray(img.astype(np.uint8)).resize((img_size[1], img_size[0]))
            if imgs.size(0) > 1:
                img.save(f'{output_file}/{i}.png')
            else:
                img.save(output_file)

