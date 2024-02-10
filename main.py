import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torchvision
from sklearn.model_selection import train_test_split
import math
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from torch.utils.data.dataloader import default_collate # 导入默认的拼接方式



class args:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_image_list_path='input/PetImages/path_train.npy'
    test_image_list_path='input/PetImages/path_test.npy'
    batch_size=128
    num_worker=8
    epoch=100
    lr=1e-4

    output_dir='output'


class ImageDataset:
    def __init__(self, image_path, transform=None):
        self.image_path_list = np.load(image_path)

        self.n_samples = len(self.image_path_list)
        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img = Image.open(self.image_path_list[idx]).convert('RGB')
        #         label = torch.tensor(self.df['outcome'][idx])
        # img = img.resize((160, 160))
        img=np.array(img)


        # print(self.transform)
        img = self.transform(img)
        return img



class Encoder(nn.Module):
    def __init__(self, do_bn=False):
        super().__init__()
        self.block1 = self.inner_block(3, 32)
        self.block2 = self.inner_block(32, 64)
        self.block3 = self.inner_block(64, 128)
        self.block4 = self.inner_block(128, 256)
        self.block5 = self.inner_block(256, 384)
        self.grayscale = transforms.Grayscale(3)

    def inner_block(self, in_c, out_c, maxpool=2):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # 3, 160, 160
        x=self.grayscale(x)
        h1 = self.block1(x)  # 32, 80, 80
        h2 = self.block2(h1)  # 64, 40, 40
        h3 = self.block3(h2)  # 128, 20, 20
        h4 = self.block4(h3)  # 256, 10, 10
        h5 = self.block5(h4)  # 384, 5, 5

        return [h1, h2, h3, h4, h5]


class Decoder(nn.Module):

    def __init__(self, do_bn):
        super().__init__()
        self.inner1 = self.inner_block(384, 256)
        self.inner2 = self.inner_block(256, 128)
        self.inner3 = self.inner_block(128, 64)
        self.inner4 = self.inner_block(64, 32)
        self.inner5 = self.inner_block(32, 3, out=True)

        self.cb1 = self.conv_block(512, 256)
        self.cb2 = self.conv_block(256, 128)
        self.cb3 = self.conv_block(128, 64)
        self.cb4 = self.conv_block(64, 32)

    def inner_block(self, in_c, out_c, out=False):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU() if not out else nn.Sigmoid(),
            nn.Dropout(0.2) if not out else nn.Identity(),
        )

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, h):
        # 384, 5, 5
        x = h[-1]
        x = self.inner1(x)  # 256, 10, 10

        x = torch.concat([x, h[-2]], dim=1)
        x = self.cb1(x)
        x = self.inner2(x)  # 128, 20, 20

        x = torch.concat([x, h[-3]], dim=1)
        x = self.cb2(x)
        x = self.inner3(x)  # 64, 40, 40

        x = torch.concat([x, h[-4]], dim=1)
        x = self.cb3(x)
        x = self.inner4(x)  # 32, 80, 80

        x = torch.concat([x, h[-5]], dim=1)
        x = self.cb4(x)
        x = self.inner5(x)  # 3, 160, 160

        return x





if __name__ == '__main__':
    print(args.device)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((160, 160)),
        transforms.Normalize(mean=0, std=1),

    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((160, 160)),
        transforms.Normalize(mean=0, std=1),

    ])

    train_dataset=ImageDataset(image_path=args.train_image_list_path,
                               transform=train_transform)
    test_dataset=ImageDataset(image_path=args.test_image_list_path,
                              transform=val_transform)

    train_loader=DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_worker,
                            shuffle=True,
                            )
    test_loader=DataLoader(test_dataset,
                           batch_size=args.batch_size,
                           num_workers=args.num_worker,
                           shuffle=False,
                           )


    encoder=Encoder(do_bn=True).to(args.device)
    decoder=Decoder(do_bn=True).to(args.device)

    # 定义多个模型参数
    model_parameters = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()},

    ]
    optimizer=torch.optim.Adam(model_parameters,lr=args.lr)
    loss_func=nn.SmoothL1Loss()


    best_loss = np.inf
    # par = tqdm(range(args.epoch))

    train_loss_list = []
    valid_loss_list = []
    print('Train..')
    for i in range(args.epoch):
        epoch_loss = 0
        epoch_loss_v = 0
        print(f'Epoch: {i + 1:02}')
        for data in tqdm(train_loader):
            data = data.to(args.device)

            optimizer.zero_grad()
            encoded_data = encoder(data)
            pre=decoder(encoded_data)
            loss = loss_func(pre, data)
            loss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss.item()




        with torch.no_grad():
            for data in tqdm(test_loader):
                data = data.to(args.device)
                encoded_data = encoder(data)
                pre = decoder(encoded_data)
                loss = loss_func(pre, data)

                epoch_loss_v = epoch_loss_v + loss.item()


        if epoch_loss_v / len(test_loader) < best_loss:
            print(f'Save Model..')
            best_loss = epoch_loss_v / len(test_loader)
            torch.save(encoder.state_dict(), f'{args.output_dir}/encoder.pt')
            torch.save(decoder.state_dict(), f'{args.output_dir}/decoder.pt')

        # print(f'Epoch: {i+1:02} | ')
        print(f'\tTrain Loss: {epoch_loss/len(train_loader):.6f} | Valid Loss: {epoch_loss_v/len(test_loader):.6}')
        # print(f'\tTrain Accuracy: {epoch_acc*100/len(train_loader):.3f}% | Valid Accuracy: {epoch_acc_v*100/len(test_loader):.3f}%')
        train_loss_list.append(epoch_loss * 100 / len(train_loader))
        valid_loss_list.append(epoch_loss_v * 100 / len(test_loader))
        np.savetxt(f'{args.output_dir}/train_loss.txt',train_loss_list,fmt='%.6f')
        np.savetxt(f'{args.output_dir}/valid_loss.txt', valid_loss_list, fmt='%.6f')
        # par.set_description_str(f'Epoch: {i + 1:02}')
        # par.set_postfix_str(
        #     f'Train Loss: {epoch_loss / len(train_loader):.6f} | Valid Loss: {epoch_loss_v  / len(test_loader):.6f}')



'''
ssh -p 41267 root@connect.westc.gpuhub.com
/InPfT9uXkh
'''