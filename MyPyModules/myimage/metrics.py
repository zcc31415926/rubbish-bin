# code based on https://github.com/juxuan27/stable-diffusion/blob/v2/ldm/metrics/quality_metrics.py
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from myimage.moco_v3 import MoCoLoss


class ImageMetrics:
    def __init__(self, device='cuda'):
        self.transforms = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.device = device
        self.evaluators = {
            'fid': self.calculateFID,
            'kid': self.calculateKID,
            'inception': self.calculateInception,
            'moco': self.calculateMoCo,
        }
        self.models_to_update = {}
        self.active_metrics = []

    @torch.no_grad()
    def initFID(self, feature=2048):
        if 'fid' in self.active_metrics:
            print('[WARNING] FID evaluator already initialized. Skipping...')
            return
        self.fid_model = FrechetInceptionDistance(feature=feature, normalize=True).to(self.device)
        self.active_metrics.append('fid')
        self.models_to_update['fid'] = self.fid_model

    @torch.no_grad()
    def initKID(self, feature=2048, subsets=100, subset_size=1000):
        if 'kid' in self.active_metrics:
            print('[WARNING] KID evaluator already initialized. Skipping...')
            return
        self.kid_model = KernelInceptionDistance(feature=feature, subsets=subsets,
                                                 subset_size=subset_size, normalize=True).to(self.device)
        self.active_metrics.append('kid')
        self.models_to_update['kid'] = self.kid_model

    @torch.no_grad()
    def initInception(self):
        if 'inception' in self.active_metrics:
            print('[WARNING] InceptionScore evaluator already initialized. Skipping...')
            return
        self.inception_model = InceptionScore(normalize=True).to(self.device)
        self.active_metrics.append('inception')

    @torch.no_grad()
    def initMoCo(self, dim=256):
        if 'moco' in self.active_metrics:
            print('[WARNING] MoCo evaluator already initialized. Skipping...')
            return
        self.moco_model = MoCoLoss(dim=dim).to(self.device)
        self.active_metrics.append('moco')

    # input: img_files - N image file path strings
    @torch.no_grad()
    def update(self, img_files, real=True, batch_size=50):
        active_update_metrics = list(set(self.active_metrics) & set(self.models_to_update.keys()))
        if len(active_update_metrics) == 0:
            print('[WARNING] No active evaluator needs update. Skipping...')
            return
        num_samples = len(img_files)
        for i in tqdm(range(num_samples // batch_size)):
            imgs = []
            for j in range(batch_size):
                img = Image.open(img_files[i * batch_size + j]).convert('RGB')
                imgs.append(self.transforms(img).unsqueeze(0))
            imgs = torch.cat(imgs, dim=0).to(self.device)
            for metric in active_update_metrics:
                self.models_to_update[metric].update(imgs, real=real)
        if num_samples % batch_size != 0:
            imgs = []
            for j in range(num_samples % batch_size):
                img = Image.open(img_files[-1 - j]).convert('RGB')
                imgs.append(self.transforms(img).unsqueeze(0))
            imgs = torch.cat(imgs, dim=0).to(self.device)
            for metric in active_update_metrics:
                self.models_to_update[metric].update(imgs, real=real)

    def calculateFID(self, imgs):
        self.fid_model.update(imgs, real=False)
        return {'  FID  ': self.fid_model.compute().item()}

    def calculateKID(self, imgs):
        self.kid_model.update(imgs, real=False)
        if len(self.kid_model.fake_features) <= self.kid_model.subset_size:
            print(f'[WARNING] The number of fake features {len(self.kid_model.fake_features)} ' +
                  f'is less than the subset size {self.kid_model.subset_size}. Skipping...')
            return {'  KID  ': -1}
        return {'  KID  ': self.kid_model.compute().item()}

    def calculateInception(self, imgs):
        self.inception_model.update(imgs)
        mean, std = self.inception_model.compute()
        return {'IS mean': mean.item(), 'IS  std': std.item()}

    def calculateMoCo(self, imgs, ref_imgs):
        moco_cos_sim = self.moco_model(imgs, ref_imgs)
        return {' MoCo  ': moco_cos_sim.mean().item()}

    # input: imgs - N*C*H*W / N*H*W*C
    # return: batch_results - dict (including all the metrics per batch)
    @torch.no_grad()
    def calculatePerBatch(self, imgs, ref_imgs=None):
        assert 'moco' not in self.active_metrics or ref_imgs is not None, \
            '[ ERROR ] MoCo evaluator requires reference images'
        if len(self.active_metrics) == 0:
            print('[WARNING] No active metric. Skipping...')
            return
        batch_results = {}
        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs)
        if imgs.shape[-1] == 3:
            imgs = imgs.permute(0, 3, 1, 2)
        imgs = imgs.to(self.device)
        for metric in self.active_metrics:
            metric_results = self.evaluators[metric](imgs)
            batch_results.update(metric_results)
        return batch_results

    # input: img_files - N image file path strings
    # return: results - dict (including all the metrics)
    @torch.no_grad()
    def calculateFromFiles(self, img_files, ref_img_files=None, batch_size=50):
        assert 'moco' not in self.active_metrics or ref_img_files is not None, \
            '[ ERROR ] MoCo evaluator requires reference image files'
        if len(self.active_metrics) == 0:
            print('[WARNING] No active metric. Skipping...')
            return
        results = None
        num_samples = len(img_files)
        ref_imgs = None
        for i in tqdm(range(num_samples // batch_size)):
            imgs = []
            for j in range(batch_size):
                img = Image.open(img_files[i * batch_size + j]).convert('RGB')
                imgs.append(self.transforms(img).unsqueeze(0))
                if ref_img_files is not None:
                    ref_img = Image.open(ref_img_files[i * batch_size + j]).convert('RGB')
                    ref_imgs.append(self.transforms(ref_img).unsqueeze(0))
            imgs = torch.cat(imgs, dim=0)
            if ref_imgs is not None:
                ref_imgs = torch.cat(ref_imgs, dim=0)
            batch_results = self.calculatePerBatch(imgs, ref_imgs)
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
                if ref_img_files is not None:
                    ref_img = Image.open(ref_img_files[i * batch_size + j]).convert('RGB')
                    ref_imgs.append(self.transforms(ref_img).unsqueeze(0))
            imgs = torch.cat(imgs, dim=0)
            if ref_imgs is not None:
                ref_imgs = torch.cat(ref_imgs, dim=0)
            batch_results = self.calculatePerBatch(imgs, ref_imgs)
            if results is None:
                results = batch_results
                for k in batch_results.keys():
                    results[k] = [results[k]]
            else:
                for k in batch_results.keys():
                    results[k].append(batch_results[k])
        return results

