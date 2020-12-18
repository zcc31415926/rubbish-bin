import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np

def draw_heatmap(width, height, x):
    fig = plt.figure()
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 1e-10)) * 255
        img = img.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        img = img[:, :, : : -1]
        plt.imshow(img)
        print('processing {}/{}'.format(i, width * height))
    plt.show()
    plt.close()

if __name__ == '__main__':
    import torchvision

    model = torchvision.models.vgg16(pretrained=True)
    model.eval()
    img_path = './imagenetsub/n01669191/ILSVRC2012_val_00000362.JPEG'
    img = cv2.imread(img_path) / 255.0
    img = cv2.resize(img, (224, 224), cv2.INTER_CUBIC)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    features = model.features(img)
    pooled_features = model.avgpool(features)
    draw_heatmap(16, 32, features.detach().numpy())
    # draw_heatmap(16, 32, pooled_features.detach().numpy())

