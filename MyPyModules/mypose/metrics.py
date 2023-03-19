# code based on https://github.com/juxuan27/stable-diffusion/blob/v2/ldm/metrics/pose_metrics.py
import os
import json
import numpy as np
import torch
from xtcocotools.coco import COCO
from mmpose.apis import inference_bottom_up_pose_model
from mmpose.apis import init_pose_model
from PIL import Image
from tqdm import tqdm

from mypose.cocosim import COCOevalSimilarity
from mypose import KptProcessor as kp


class PoseMetrics:
    def __init__(self, device='cuda'):
        self.device = device
        self.mmpose = False
        self.evaluators = {
            'mmpose': self.calculateMMPOSE,
        }
        self.active_metrics = []

    @torch.no_grad()
    def initMMPOSE(self):
        if 'mmpose' in self.active_metrics:
            print('[WARNING] MMPOSE evaluator already initialized. Skipping...')
            return
        ckpt_path = os.path.join(os.path.dirname(__file__), 'mmpose_ckpt.pth')
        err_log = '[ ERROR ] There has to be a valid MMPose weight file ' + \
            'named `mmpose_ckpt.pth` in the root directory. Please check the MMPose docs: ' + \
            'https://github.com/open-mmlab/mmpose/blob/master/docs/en/install.md for more information'
        assert os.path.exists(ckpt_path), err_log
        self.mmpose_model = init_pose_model(
            os.path.join(os.path.dirname(__file__), 'mmpose_config.py'),
            ckpt_path, device=self.device,
        )
        self.mmpose_model.eval()
        self.tmp_res_dir = 'out/posemetrics_tmp_results'
        self.active_metrics.append('mmpose')

    def calculateMMPOSE(self, imgs, poses):
        if not os.path.exists(self.tmp_res_dir):
            os.makedirs(self.tmp_res_dir)
        n, h, w, c = imgs.shape
        gt_pose_results = {
            'images': [],
            'annotations': [],
            'categories': [{'id': 1, 'name': 'person'}]
        }
        dt_pose_results = []
        for idx in range(n):
            gt_pose_results['images'].append({
                'file_name': 'None', 'id': idx, 'page_url': 'None', 'image_url': 'None',
                'height': h, 'width': w,
                'picture_name': 'None', 'author': 'None', 'description': 'None', 'category': 'None',
            })
            for anno_i in range(poses.shape[1]):
                present_annotation = poses[idx, anno_i]
                keypoint_num = (present_annotation[..., 0] >= 0).astype(np.int32).sum()
                if keypoint_num > 0:
                    gt_pose_results['annotations'].append({
                        'keypoints': list(present_annotation.reshape(-1)),
                        'num_keypoints': int(keypoint_num),
                        'iscrowd': 0,
                        'image_id': idx,
                        'category_id': 1,
                        'id': idx * 10 + anno_i,
                        'bbox': [
                            present_annotation[:, 0].min(), present_annotation[:, 1].min(),
                            present_annotation[:, 0].max() - present_annotation[:, 0].min(),
                            present_annotation[:, 1].max() - present_annotation[:, 1].min(),
                        ],
                        'area': (present_annotation[:, 1].max() - present_annotation[:, 1].min()) * \
                            (present_annotation[:, 0].max() - present_annotation[:, 0].min()),
                    })
            present_image = imgs[idx].copy()
            if (present_image <= 1).all():
                present_image = (present_image * 255).astype(np.uint8)
            pose_results, _ = inference_bottom_up_pose_model(
                self.mmpose_model, present_image, pose_nms_thr=1)
            if len(pose_results) > 0:
                for pose_result in pose_results:
                    dt_pose_results.append({
                        'category_id': 1,
                        'image_id': idx,
                        'keypoints': [content.item() for content in pose_result['keypoints'].reshape(-1)],
                        'score': pose_result['score'].item(),
                    })
        gt_file_path = os.path.join(self.tmp_res_dir, 'gt_keypoints.json')
        with open(gt_file_path, 'w') as f:
            json.dump(gt_pose_results, f)
        dt_file_path = os.path.join(self.tmp_res_dir, 'dt_keypoints.json')
        with open(dt_file_path, 'w') as f:
            json.dump(dt_pose_results, f)
        gt_coco = COCO(gt_file_path)
        dt_coco = gt_coco.loadRes(dt_file_path)
        coco_eval = COCOevalSimilarity(gt_coco, dt_coco, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        ap_ar_results = {
            'Distance AP @ IoU=0.50:0.95 | area=   all | maxDets=20': coco_eval.stats[0],
            'Distance AP @ IoU=0.50      | area=   all | maxDets=20': coco_eval.stats[1],
            'Distance AP @ IoU=0.75      | area=   all | maxDets=20': coco_eval.stats[2],
            'Distance AP @ IoU=0.50:0.95 | area=medium | maxDets=20': coco_eval.stats[3],
            'Distance AP @ IoU=0.50:0.95 | area= large | maxDets=20': coco_eval.stats[4],
            'Distance AR @ IoU=0.50:0.95 | area=   all | maxDets=20': coco_eval.stats[5],
            'Distance AR @ IoU=0.50      | area=   all | maxDets=20': coco_eval.stats[6],
            'Distance AR @ IoU=0.75      | area=   all | maxDets=20': coco_eval.stats[7],
            'Distance AR @ IoU=0.50:0.95 | area=medium | maxDets=20': coco_eval.stats[8],
            'Distance AR @ IoU=0.50:0.95 | area= large | maxDets=20': coco_eval.stats[9],
        }
        coco_eval.evaluateSimilarity()
        coco_eval.accumulateSimilarity()
        coco_eval.summarizeSimilarity()
        cosine_similarity_results={
            'Similarity AP @ IoU=0.50:0.95 | area=   all | maxDets=20': coco_eval.statsSimilarity[0],
            'Similarity AP @ IoU=0.50      | area=   all | maxDets=20': coco_eval.statsSimilarity[1],
            'Similarity AP @ IoU=0.75      | area=   all | maxDets=20': coco_eval.statsSimilarity[2],
            'Similarity AP @ IoU=0.50:0.95 | area=medium | maxDets=20': coco_eval.statsSimilarity[3],
            'Similarity AP @ IoU=0.50:0.95 | area= large | maxDets=20': coco_eval.statsSimilarity[4],
            'Similarity AR @ IoU=0.50:0.95 | area=   all | maxDets=20': coco_eval.statsSimilarity[5],
            'Similarity AR @ IoU=0.50      | area=   all | maxDets=20': coco_eval.statsSimilarity[6],
            'Similarity AR @ IoU=0.75      | area=   all | maxDets=20': coco_eval.statsSimilarity[7],
            'Similarity AR @ IoU=0.50:0.95 | area=medium | maxDets=20': coco_eval.statsSimilarity[8],
            'Similarity AR @ IoU=0.50:0.95 | area= large | maxDets=20': coco_eval.statsSimilarity[9],
        }
        human_number_difference = []
        for img_idx in range(len(gt_coco.imgToAnns)):
            human_number_difference.append(abs(len(gt_coco.imgToAnns[img_idx]) -
                                               len(dt_coco.imgToAnns[img_idx])))
        human_number_difference_results = {
            'Human Number Difference': np.mean(human_number_difference),
        }
        os.system(f'rm -rf {self.tmp_res_dir}')
        return {**ap_ar_results, **cosine_similarity_results, **human_number_difference_results}

    # input: imgs - N*H*W*C
    # input: poses - N*K*17*3 (max number of persons: K)
    # return: batch_results - dict (including all the metrics per batch)
    @torch.no_grad()
    def calculatePerBatch(self, imgs, poses):
        if len(self.active_metrics) == 0:
            print('[WARNING] no active metric. Skipping...')
            return
        batch_results = {}
        if isinstance(poses, torch.Tensor):
            poses = poses.cpu().numpy()
        elif isinstance(poses, list):
            poses = np.array(poses)
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.cpu().numpy()
        for metric in self.active_metrics:
            metric_results = self.evaluators[metric](imgs, poses)
            batch_results.update(metric_results)
        return batch_results

    # input: img_files - N image file path strings
    # input: pose_files - N pose file path strings
    # input: img_size - (H, W)
    # return: results - dict (including all the metrics)
    @torch.no_grad()
    def calculateFromFiles(self, img_files, pose_files, pose_mode='openpose18',
                           pose_img_sizes=None, batch_size=50):
        if len(self.active_metrics) == 0:
            print('[WARNING] no active metric. Skipping...')
            return
        results = None
        num_samples = len(pose_files)
        for i in tqdm(range(num_samples // batch_size)):
            imgs = []
            poses = []
            for j in range(batch_size):
                img = np.array(Image.open(img_files[i * batch_size + j]))
                img_size = img.shape[: 2]
                imgs.append(img)
                pose = kp.loadPose(pose_files[i * batch_size + j], in_mode=pose_mode, keep_score=True)
                if (pose > 1).any():
                    pose = kp.poseCoordinates2Ratio(pose, pose_img_sizes[j])
                poses.append(kp.fitPoseInImg(pose, img_size))
            imgs = np.array(imgs)
            poses = np.array(poses)
            batch_results = self.calculatePerBatch(imgs, poses)
            if results is None:
                results = batch_results
                for k in batch_results.keys():
                    results[k] = [results[k]]
            else:
                for k in batch_results.keys():
                    results[k].append(batch_results[k])
        if num_samples % batch_size != 0:
            imgs = []
            poses = []
            for j in range(batch_size):
                img = np.array(Image.open(img_files[i * batch_size + j]))
                img_size = img.shape[: 2]
                imgs.append(img)
                pose = kp.loadPose(pose_files[i * batch_size + j], in_mode=pose_mode, keep_score=True)
                if (pose > 1).any():
                    pose = kp.poseCoordinates2Ratio(pose, pose_img_sizes[j])
                poses.append(kp.fitPoseInImg(pose, img_size))
            imgs = np.array(imgs)
            poses = np.array(poses)
            batch_results = self.calculatePerBatch(imgs, poses)
            if results is None:
                results = batch_results
                for k in batch_results.keys():
                    results[k] = [results[k]]
            else:
                for k in batch_results.keys():
                    results[k].append(batch_results[k])
        return results

