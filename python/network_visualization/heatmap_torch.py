import cv2
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision.models import vgg16, VGG16_Weights


def draw_heatmap(features, num_rows, num_cols, map_size=(16, 16), margin=0.125):
    assert len(features) == num_rows * num_cols, \
        f'incompatible numbers of features {len(features)}, rows {num_rows} and columns {num_cols}'
    vis = []
    with tqdm(range(num_rows * num_cols)) as progress:
        for i in progress:
            f = features[i]
            f = (f - f.min()) / (f.max() - f.min() + 1e-10)
            f = (f * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(f, cv2.COLORMAP_JET)
            vis.append(cv2.resize(heatmap, map_size))
    vis = np.array(vis).reshape(num_rows, num_cols, *map_size, 3)
    rows = []
    for row in vis:
        new_row = []
        for i in range(len(row) - 1):
            blank_img = np.ones((map_size[0], int(map_size[1] * (1 + margin)), 3))
            blank_img[:, : map_size[1]] = row[i]
            new_row.append(blank_img)
        new_row.append(row[-1])
        rows.append(np.hstack(new_row))
    for i in range(len(rows) - 1):
        blank_img = np.ones((int(map_size[0] * (1 + margin)),
                             int(map_size[1] * (1 + margin)) * (num_cols - 1) + map_size[1], 3))
        blank_img[: map_size[0]] = rows[i]
        rows[i] = blank_img
    vis = np.vstack(rows)
    return vis


if __name__ == '__main__':
    img_path = '/mnt/d/Pictures/warframe.jpg'
    img_size = (224, 224)
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).float()
    features = model.features(img)
    vis = draw_heatmap(features[0].detach().cpu().numpy(), num_rows=16,
                       num_cols=32, map_size=(32, 32), margin=0.125)
    cv2.imshow('vis.png', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

