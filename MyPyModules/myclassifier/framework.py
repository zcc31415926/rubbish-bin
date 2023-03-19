import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from myclassifier.netutils import Classifier


class Trainer:
    def __init__(self, net, checkpoint=None, device='cuda'):
        self.classifier = Classifier(net)
        checkpoint = checkpoint if checkpoint is not None else os.path.abspath('./ckpt.pth')
        ckpt_dir = os.path.dirname(checkpoint)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        infix = '-new' if os.path.exists(checkpoint) else ''
        self.saved_checkpoint = f'{"".join(checkpoint.split(".")[: -1])}{infix}.{checkpoint.split(".")[-1]}'
        self.device = device
        self.start_from = 0
        self.max_acc = 0
        if os.path.exists(checkpoint):
            recordings = torch.load(checkpoint, map_location='cpu')
            if 'epoch' in recordings.keys():
                self.classifier.load_state_dict(recordings['state_dict'])
                self.start_from = recordings['epoch'] + 1
                self.max_acc = recordings['acc']
            else:
                self.classifier.load_state_dict(recordings)
        self.classifier = self.classifier.to(self.device)
        self.train_loader = None
        self.val_loader = None

    def initDataLoader(self, DATASET, data_dir, img_size=None, batch_size=16,
                       mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], train=True, val=True):
        if train:
            self.train_loader = self.getDataLoader(DATASET, data_dir, img_size,
                                                   batch_size, mean, std, train=True)
        if val:
            self.val_loader = self.getDataLoader(DATASET, data_dir, img_size,
                                                 batch_size, mean, std, train=False)

    def getDataLoader(self, DATASET, data_dir, img_size=None, batch_size=16,
                      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], train=True):
        identity = transforms.Lambda(lambda x: x)
        t = transforms.Compose([
            transforms.RandomHorizontalFlip() if train else identity,
            transforms.Resize(img_size) if img_size is not None else identity,
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = DATASET(root=data_dir, train=train, download=True, transform=t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train,
                                num_workers=4, drop_last=True, pin_memory=True)
        return dataloader

    def train(self, lr, num_epochs):
        assert self.train_loader is not None, '[ ERROR ] Train loader must be initialized'
        self.classifier.train()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        for e in range(self.start_from, num_epochs):
            with tqdm(self.train_loader, dynamic_ncols=True) as loader_tqdm:
                for x, y in loader_tqdm:
                    optimizer.zero_grad()
                    x = x.to(self.device)
                    y = y.to(self.device)
                    loss = self.classifier.loss(x, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 1)
                    optimizer.step()
                    loader_tqdm.set_postfix(ordered_dict={
                        'epoch': f'{e}/{num_epochs}',
                        'loss': loss.item(),
                        'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                    })
                    loader_tqdm.set_description('training')
            if e % 5 == 0:
                acc = self.val()
                recordings = {
                    'state_dict': self.classifier.state_dict(),
                    'epoch': e,
                    'acc': acc,
                }
                torch.save(recordings, self.saved_checkpoint)
                self.classifier.train()
                if acc > self.max_acc:
                    self.max_acc = acc
                    torch.save(recordings, f'{self.saved_checkpoint}.best')

    @torch.no_grad()
    def val(self):
        assert self.val_loader is not None, '[ ERROR ] Val loader must be initialized'
        self.classifier.eval()
        num_correct = 0
        with tqdm(self.val_loader, dynamic_ncols=True) as loader_tqdm:
            for x, y in loader_tqdm:
                x = x.to('cuda')
                y = y.to('cuda')
                predicted_y = self.classifier(x)
                predicted_idx = torch.argmax(predicted_y, dim=-1)
                num_correct += (predicted_idx == y).float().sum().item()
                loader_tqdm.set_description('validation')
        acc = 100 * num_correct / len(self.val_loader) / self.val_loader.batch_size
        print(f'acc: {acc}')
        return acc

    def extractStateDict(self, in_file, out_file):
        recordings = torch.load(in_file, map_location='cpu')
        assert 'epoch' in recordings.keys(), '[ ERROR ] Input file format not supported'
        state_dict = recordings['state_dict']
        torch.save(state_dict, out_file)
        print(f'[  LOG  ] Model weights saved in {out_file}')

