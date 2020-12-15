# class activation mapping

import cv2
import torch
import numpy as np

def tmp(x):
    global gradients
    gradients = x

def draw_cam(img_path, model, cam_path, cammed_img_path, img_size=(224, 224), cam_ratio=0.5):
    model.eval()
    img = cv2.imread(img_path)
    img_input = cv2.resize(img, img_size, cv2.INTER_CUBIC) / 255.0
    img_input = img_input.transpose(2, 0, 1)
    img_input = np.expand_dims(img_input, 0)
    img_input = torch.from_numpy(img_input).float()
    # img_input.requires_grad_(True)
    # img_input.register_hook(tmp)

    ########## model definition ##########
    # input: img_input
    # output: features, output
    # structure: encoder, view, decoder
    features = model.features(img_input)
    pooled_features = model.avgpool(features)
    flattened_features = pooled_features.view(-1, 25088)
    output = model.classifier(flattened_features)
    ########## model definition ##########

    pred_index = torch.argmax(output[0]).item()
    pred_score = output[0, pred_index]
    features.register_hook(tmp)
    pred_score.backward()
    grad = gradients[0]
    importance_factor = torch.nn.functional.adaptive_avg_pool2d(grad, (1, 1))
    grad *= importance_factor
    grad = torch.mean(grad, 0)
    cam_map = grad.numpy()
    cam_map = np.abs(cam_map)
    cam_map -= np.min(cam_map)
    cam_map /= np.max(cam_map)

    cam_map = cv2.resize(cam_map, (np.shape(img)[1], np.shape(img)[0]), cv2.INTER_CUBIC)
    cam_map = np.uint8(255 * cam_map)
    # blue for 0s (least important); red for 1s (most important)
    cam_map = cv2.applyColorMap(cam_map, cv2.COLORMAP_JET)
    cammed_img = cam_map * cam_ratio + img * (1 - cam_ratio)
    cv2.imwrite(cam_path, cam_map)
    cv2.imwrite(cammed_img_path, cammed_img)

if __name__ == '__main__':
    import torchvision

    model = torchvision.models.vgg16(pretrained=True)
    img_path = './imagenetsub/n01669191/ILSVRC2012_val_00001662.JPEG'
    draw_cam(img_path, model, './cam_map.jpg', './cammed_img.jpg', (224, 224))

