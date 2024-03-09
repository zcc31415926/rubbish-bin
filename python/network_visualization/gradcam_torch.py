# gradient class activation mapping (Grad-CAM) and Grad-CAM++
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from torch.autograd import Function
from PIL import Image
import cv2
import torch
import numpy as np


class GuidedReLU(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.nn.functional.relu(x)

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        back_mask = (grad >= 0).float()
        return grad * (x >= 0).float() * back_mask


class GradCAM:
    def __init__(self, img_size=(224, 224), pixel_weight=False, plus=True, guided_relu=False):
        self.model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.model.eval()
        if guided_relu:
            self.modify_model()
        print('model:')
        print(self.model)
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.pixel_weight = pixel_weight
        self.plus = plus
        self.guided_relu = guided_relu
        self.grad_feats = None

    def modify_model(self):
        def modify_module(module):
            for name, submodule in module._modules.items():
                if len(submodule._modules.items()) > 0:
                    modify_module(submodule)
                elif submodule.__class__.__name__ == 'ReLU':
                    module._modules[name] = GuidedReLU.apply
        modify_module(self.model)

    def record_grad(self, x):
        self.grad_feats = x

    def run(self, img, target_idx):
        img.requires_grad_(True)
        # VGG-specific
        features = self.model.features(img)
        pooled_features = self.model.avgpool(features)
        flattened_features = pooled_features.view(-1, 25088)
        output = self.model.classifier(flattened_features)
        if target_idx is None:
            target_index = torch.argmax(output[0]).item()
        print(f'target index: {target_index}')
        pred_score = output[0, target_index]
        features.register_hook(self.record_grad)
        pred_score.backward()
        return features.detach().cpu(), img.grad.cpu().data

    def draw_cam(self, img_path, cam_path, cam_img_path, grad_path, target_idx=None, cam_ratio=0.5):
        img = Image.open(img_path).convert('RGB')
        img_input = self.transform(img).unsqueeze(0).float()
        img = np.array(img)
        features, grad_input = self.run(img_input, target_idx)
        if self.plus:
            features_sum = torch.sum(features, dim=(2, 3), keepdim=True)
            alpha = self.grad_feats ** 2 / (2 * self.grad_feats ** 2 +
                                            features_sum * self.grad_feats ** 3 + 1e-10)
            self.grad_feats *= alpha
        weight = torch.nn.functional.relu(self.grad_feats)
        if not self.pixel_weight:
            weight = torch.mean(weight, dim=(2, 3), keepdim=True)
        cam = torch.mean(features * weight, dim=(0, 1))
        cam = cam.detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam -= np.min(cam)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (np.shape(img)[1], np.shape(img)[0]))
        cam = np.uint8(255 * cam)
        # blue for 0s (least important); red for 1s (most important)
        cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        cam_img = cam * cam_ratio + img * (1 - cam_ratio)
        cv2.imwrite(cam_path, cam)
        cv2.imwrite(cam_img_path, cam_img)
        grad_input = torch.mean(grad_input, dim=(0, 1))
        grad_input = grad_input.detach().cpu().numpy()
        grad_input = np.maximum(grad_input, 0)
        grad_input -= np.min(grad_input)
        grad_input /= np.max(grad_input)
        grad_input = cv2.resize(grad_input, (np.shape(img)[1], np.shape(img)[0]))
        grad_input = np.uint8(255 * grad_input)
        # blue for 0s (least important); red for 1s (most important)
        cv2.imwrite(grad_path, grad_input)


if __name__ == '__main__':
    img_path = '/mnt/d/Pictures/warframe.jpg'
    grad_cam = GradCAM(img_size=(224, 224), pixel_weight=False, plus=False, guided_relu=True)
    grad_cam.draw_cam(img_path, cam_path='./cam.jpg', cam_img_path='./cam_img.jpg',
                      grad_path='./grad.jpg', cam_ratio=0.5)

