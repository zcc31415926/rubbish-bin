import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import seaborn as sns

matplotlib.use('Agg')
sns.set()


class KptProcessor:
    mode_kpts = {
        'coco': 17,
        'openpose18': 18,
    }
    transforms = {
        'coco_openpose18': np.array([0, 17, 5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16, 1, 2, 3, 4]),
    }

    # COCO data annotations:
    # 0: nose;
    # 1: left eye; 2: right eye; 3: left ear; 4: right ear
    # 5: left shoulder; 6: right shoulder; 7: left elbow; 8: right elbow
    # 9: left wrist; 10: right wrist; 11: left hip; 12: right hip
    # 13: left knee; 14: right knee; 15: left ankle; 16: right ankle
    # 17: neck
    @classmethod
    def loadPose(cls, pose_data, in_mode='coco', out_mode='coco', keep_score=False):
        assert in_mode in KptProcessor.mode_kpts.keys(), f'[ ERROR ] Input mode {in_mode} not supported'
        assert out_mode in KptProcessor.mode_kpts.keys(), f'[ ERROR ] Output mode {out_mode} not supported'
        if isinstance(pose_data, list) or isinstance(pose_data, np.ndarray):
            pose = np.array(pose_data)
        elif pose_data.endswith('txt'):
            pose = np.loadtxt(pose_data)
        else:
            print(f'[ ERROR ] pose_data format {type(pose_data)} not supported')
            raise NotImplementedError
        pose = pose.reshape(-1, KptProcessor.mode_kpts[in_mode], pose.shape[-1])
        if not keep_score:
            pose = pose[..., : 2]
        for i in range(pose.shape[0]):
            for j in range(pose.shape[1]):
                if pose[i, j, 0] < 0 or pose[i, j, 1] < 0:
                    pose[i, j, 0] = -1
                    pose[i, j, 1] = -1
                    if pose.shape[2] > 2:
                        pose[i, j, 2] = -1
        if in_mode == 'coco' and out_mode == 'openpose18':
            negative_mask = (pose[:, 5 : 6] < 0).astype(np.float32) + (pose[:, 6 : 7] < 0).astype(np.float32)
            neck = (pose[:, 5 : 6] + pose[:, 6 : 7]) / 2
            neck[negative_mask.astype(np.bool8)] = -1
            pose = np.append(pose, neck, axis=1)
        if in_mode != out_mode:
            if f'{in_mode}_{out_mode}' in KptProcessor.transforms.keys():
                seq = KptProcessor.transforms[f'{in_mode}_{out_mode}']
            else:
                seq = KptProcessor.transforms[f'{out_mode}_{in_mode}'].argsort()
            pose = pose[:, seq[: KptProcessor.mode_kpts[out_mode]]]
        return pose

    @classmethod
    def poseCoordinates2Ratio(cls, pose, img_size):
        pose = pose.astype(np.float64)
        pose[..., : 2] /= max(img_size)
        pose[pose >= 1] = 1 - 1e-10
        return pose

    # [WARNING] the adjusted pose may have inconsistent truncated keypoints
    @classmethod
    def fitPoseInImg(cls, pose, img_size):
        assert (pose <= 1).all(), '[ ERROR ] The pose data should be in [H, W] ratio form'
        pose[..., : 2] *= min(img_size)
        if img_size[0] < img_size[1]:
            pose[..., 0][pose[..., 0] >= 0] += 0.5 * (img_size[1] - img_size[0])
        else:
            pose[..., 1][pose[..., 1] >= 0] += 0.5 * (img_size[0] - img_size[1])
        pose[pose < 0] = -1
        return pose


class KptDrawer:
    skeleton = [
        [0, 1],   [0, 2],   [1, 3],   [2, 4],
        [3, 5],   [4, 6],   [5, 7],   [6, 8],
        [7, 9],   [8, 10],  [5, 11],  [6, 12],
        [11, 13], [12, 14], [13, 15], [14, 16],
    ]
    color = sns.color_palette('hls', len(skeleton))

    @classmethod
    def draw(cls, pose, img_size, output_file, draw_individual=False, background_color='white'):
        pose = np.array(pose).astype(np.int32)
        if len(pose.shape) == 2:
            pose = np.expand_dims(pose, axis=0)
        assert pose.shape[1] == 17, '[ ERROR ] only coco pose format supported'
        global_img = Image.new('RGB', (img_size[1], img_size[0]), color=background_color)
        person_imgs = []
        for person in pose:
            global_drawer = ImageDraw.Draw(global_img)
            for i, [pa, pb] in enumerate(KptDrawer.skeleton):
                line_color = tuple([int(255 * c) for c in KptDrawer.color[i]])
                if person[pa, 0] >= 0 and person[pa, 1] >= 0 and \
                    person[pb, 0] >= 0 and person[pb, 1] >= 0:
                    global_drawer.line([(person[pa, 0], person[pa, 1]),
                                        (person[pb, 0], person[pb, 1])], line_color, width=5)
            for i, point in enumerate(person):
                global_drawer.text(point, str(i))
            if draw_individual:
                person_img = Image.new('RGB', (img_size[1], img_size[0]), color='black')
                person_drawer = ImageDraw.Draw(person_img)
                for i, [pa, pb] in enumerate(KptDrawer.skeleton):
                    line_color = tuple([int(255 * c) for c in KptDrawer.color[i]])
                    if person[pa, 0] >= 0 and person[pa, 1] >= 0 and \
                        person[pb, 0] >= 0 and person[pb, 1] >= 0:
                        person_drawer.line([(person[pa, 0], person[pa, 1]),
                                            (person[pb, 0], person[pb, 1])], line_color, width=5)
                for i, point in enumerate(person):
                    person_drawer.text(point, str(i))
                person_imgs.append(np.array(person_img))
        global_img.save(output_file)
        if draw_individual:
            for i, img in enumerate(person_imgs):
                img.save(f'{output_file[: -4]}_person{i}.png')
        print(f'[  LOG  ] Pose images saved in {output_file}')
        if draw_individual:
            print(f'[  LOG  ] Individual pose images saved in {output_file[: -4]}_personX.png')

