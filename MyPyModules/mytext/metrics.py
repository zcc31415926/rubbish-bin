# code based on https://github.com/juxuan27/stable-diffusion/blob/v2/ldm/metrics/text_metrics.py
import numpy as np
import torch
from torchmetrics.multimodal import CLIPScore
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


class TextMetrics:
    def __init__(self, device):
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.device = device
        self.evaluators = {
            'clipsim': self.calculateCLIPSIM,
        }
        self.active_metrics = []

    @torch.no_grad()
    def initCLIP(self, clip_model='openai/clip-vit-base-patch16'):
        if 'clipsim' in self.active_metrics:
            print('[WARNING] FID evaluator already initialized. Skipping...')
            return
        self.clip_model = CLIPScore(model_name_or_path=clip_model).to(self.device)
        self.active_metrics.append('clipsim')

    def calculateCLIPSIM(self, imgs, txts):
        return {'CLIPSIM': self.clip_model(imgs, txts).item()}

    @torch.no_grad()
    def calculatePerBatch(self, imgs, txts):
        if len(self.active_metrics) == 0:
            print('[WARNING] No active metric. Skipping...')
            return
        batch_results = {}
        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs)
        if imgs.shape[-1] == 3:
            imgs = imgs.permute(0, 3, 1, 2)
        if (imgs > 1).any():
            imgs /= 255.0
        imgs = imgs.to(self.device)
        for metric in self.active_metrics:
            metric_results = self.evaluators[metric](imgs, txts)
            batch_results.update(metric_results)
        return batch_results

    @torch.no_grad()
    def calculateFromFiles(self, img_files, txts, batch_size=50):
        if len(self.active_metrics) == 0:
            print('[WARNING] No active metric. Skipping...')
            return
        results = None
        num_samples = len(img_files)
        for i in tqdm(range(num_samples // batch_size)):
            imgs = []
            for j in range(batch_size):
                img = Image.open(img_files[i * batch_size + j]).convert('RGB')
                imgs.append(self.transforms(img).unsqueeze(0))
            imgs = torch.cat(imgs, dim=0)
            batch_results = self.calculatePerBatch(imgs, txts[i * batch_size : (i + 1) * batch_size])
            if results is None:
                results = batch_results
                for k in batch_results.keys():
                    results[k] = [results[k]]
            else:
                for k in batch_results.keys():
                    results[k].append(batch_results[k])
        if num_samples % batch_size != 0:
            imgs = []
            for j in range(num_samples % batch_size):
                img = Image.open(img_files[-1 - j]).convert('RGB')
                imgs.append(self.transforms(img).unsqueeze(0))
            imgs = torch.cat(imgs, dim=0)
            batch_results = self.calculatePerBatch(imgs, txts[-1 : -1 - num_samples % batch_size : -1])
            if results is None:
                results = batch_results
                for k in batch_results.keys():
                    results[k] = [results[k]]
            else:
                for k in batch_results.keys():
                    results[k].append(batch_results[k])
        return results

