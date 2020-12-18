import torch
import torchvision
import torch.nn as nn
from torch.autograd import Function
import cv2
import numpy as np

class GuidedReLU(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return nn.functional.relu(x)

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        back_mask = (grad >= 0).float()
        return grad * (x >= 0).float() * back_mask

class WrappedConv(nn.Module):
    features = None
    grad = None

    def __init__(self, conv):
        super(WrappedConv, self).__init__()
        self.conv = conv

    def forward(self, x):
        y = self.conv(x)
        WrappedConv.features = y[0]
        y.register_hook(self.record_grad)
        return y

    def record_grad(self, grad):
        if WrappedConv.grad is None:
            WrappedConv.grad = grad.detach().cpu().numpy()[0]

class GradCamGenerator:
    def __init__(self, model, img_path, img_size, target_index):
        self.model = model
        self.img_size = img_size
        self.x = self.process_img(img_path).requires_grad_(True)
        self.target_index = target_index

    def process_img(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_size) / 255.0
        img = np.expand_dims(img.transpose(2, 0, 1), 0)
        return torch.from_numpy(img).float()

    def modify_network(self):
        def modify_module(module):
            for name, submodule in module._modules.items():
                if len(submodule._modules.items()) > 0:
                    modify_module(submodule)
                elif submodule.__class__.__name__ == 'ReLU':
                    module._modules[name] = GuidedReLU.apply
                elif submodule.__class__.__name__ == 'Conv2d':
                    module._modules[name] = WrappedConv(submodule)
        modify_module(self.model)

    def run_forward_and_backward(self):
        self.modify_network()
        # print(self.model)
        y = self.model(self.x)
        if self.target_index is None:
            self.target_index = torch.argmax(y[0])
        target_value = y[0, self.target_index]
        target_value.backward()
        return self.x.grad.cpu().data.numpy()[0]

    def generate_cam(self):
        grad_input = self.run_forward_and_backward()
        assert WrappedConv.features is not None
        assert WrappedConv.grad is not None
        features = WrappedConv.features.detach().cpu().numpy()
        features *= np.mean(WrappedConv.grad, axis=(1, 2), keepdims=True)
        features = np.sum(features, axis=0)
        features = np.maximum(features, 0)
        features = features - np.min(features)
        features = features / np.max(features)
        features = cv2.resize(features, self.img_size)
        grad_input = grad_input.transpose(1, 2, 0)
        features = np.array([features for i in range(self.x.size(1))]).transpose(1, 2, 0)
        cam = grad_input * features
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        grad_input = grad_input - np.min(grad_input)
        grad_input = grad_input / np.max(grad_input)
        return (grad_input * 255.0).astype(np.uint8), (cam * 255.0).astype(np.uint8)

if __name__ == '__main__':
    model = torchvision.models.resnet50(pretrained=True)
    model.eval()
    img_path = '../data_0_0_recaptured.jpg'
    img_size = (224, 224)
    grad_cam_generator = GradCamGenerator(model, img_path, img_size, None)
    gb, gb_cam = grad_cam_generator.generate_cam()
    cv2.imwrite('gb.png', gb)
    cv2.imwrite('gb_cam.png', gb_cam)
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, img_size)
    cv2.imwrite('original_img.png', original_img)

