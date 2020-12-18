# gradient class activation mapping ++

import cv2
import torch
import numpy as np

def tmp(x):
    global gradients
    gradients = x

def normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for i in range(3):
        img_slice = img[..., i]
        img_slice -= np.mean(img_slice)
        img_slice /= np.std(img_slice)
        img[..., i] = img_slice * std[i] + mean[i]
    return img

def draw_cam(img_path, model, cam_path, cammed_img_path, img_size=(224, 224), cam_ratio=0.5):
    model.eval()
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = cv2.resize(img, img_size, cv2.INTER_CUBIC) / 255.0
    img_input = normalize(img_input)
    img_input = img_input.transpose(2, 0, 1)
    img_input = np.expand_dims(img_input, 0)
    img_input = torch.from_numpy(img_input).float()

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
    print('prediction index', pred_index)
    pred_score = output[0, pred_index]
    features.register_hook(tmp)
    pred_score.backward()
    grad = gradients[0]

    features_sum = torch.sum(features[0], (1, 2), keepdim=True)
    alpha = grad ** 2 / (2 * grad ** 2 + features_sum * grad ** 3 + 1e-10)
    positive_grad = torch.nn.functional.relu(grad)
    weight = torch.sum(alpha * positive_grad, (1, 2), keepdim=True)
    cam_map = torch.sum(weight * features[0], 0)

    cam_map = cam_map.detach().cpu().numpy()
    cam_map = np.maximum(cam_map, 0)
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
    img_path = '/home/charlie/Pictures/rosmt2.png'
    draw_cam(img_path, model, './cam_map.jpg', './cammed_img.jpg', (224, 224))

