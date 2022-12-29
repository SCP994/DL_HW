import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from SimpleMobileNet import SimpleMobileNet
from SimpleNet import SimpleNet
from ResNet18 import res_model


class BitmojiData(Dataset):
    def __init__(self, csv_path, img_path, mode='train', test_ratio=0.1, resize_height=224, resize_width=224):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            mode (string): 训练模式还是测试模式
            test_ratio (float): 测试集比例
        """

        self.data_info = pd.read_csv(csv_path)  # 读取 csv 文件
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.img_path = img_path
        self.mode = mode
        self.data_len = len(self.data_info.index)
        self.train_len = int(self.data_len * (1 - test_ratio))

        if mode == 'train':
            # 第一列是图像文件的名称，第二列是标签
            self.train_image = np.asarray(self.data_info.iloc[:self.train_len, 0])
            self.train_label = np.asarray(self.data_info.iloc[:self.train_len, 1])
            self.image_arr = self.train_image
            self.label_arr = self.train_label
        elif mode == 'test':
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len:, 0])
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:, 1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'val':
            self.test_image = np.asarray(self.data_info.iloc[:, 0])
            self.image_arr = self.test_image

        self.real_len = len(self.image_arr)
        print('Finished reading the {} set of Bitmoji Dataset ({} samples found)'.format(mode, self.real_len))

    def __getitem__(self, index):
        # 从 image_arr 中得到索引对应的文件名
        img_name = self.image_arr[index]

        # 读取图像文件
        img = Image.open(self.img_path + img_name)
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.resize_height, self.resize_width)),
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
                transforms.ToTensor(),
                transforms.Normalize((0.7856, 0.7355, 0.6979), (0.3438, 0.3537, 0.3752))  # 归一化
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.resize_height, self.resize_width)),
                transforms.ToTensor(),
                transforms.Normalize((0.7856, 0.7355, 0.6979), (0.3438, 0.3537, 0.3752))
            ])
        img = transform(img)

        if self.mode == 'val':
            return img
        else:
            label = self.label_arr[index]
            label = 0 if label == -1 else 1
            return img, label

    def __len__(self):
        return self.real_len


if __name__ == '__main__':
    # 标签路径和图像路径
    shuffled_train_csv_path = 'C:/Users/Ohh/Desktop/pythonProject/shuffled_train.csv'
    train_img_path = 'C:/Users/Ohh/Desktop/bitmoji-faces-gender-recognition/BitmojiDataset/trainimages/'
    val_csv_path = 'C:/Users/Ohh/Desktop/bitmoji-faces-gender-recognition/sample_submission.csv'
    val_img_path = 'C:/Users/Ohh/Desktop/bitmoji-faces-gender-recognition/BitmojiDataset/testimages/'
    train_dataset = BitmojiData(shuffled_train_csv_path, train_img_path, mode='train')
    test_dataset = BitmojiData(shuffled_train_csv_path, train_img_path, mode='test')
    val_dataset = BitmojiData(val_csv_path, val_img_path, mode='val')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=3,
        shuffle=True,
        num_workers=8
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=3,
        shuffle=False,
        num_workers=8
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 超参数
    learning_rate = 3e-4
    weight_decay = 1e-3
    num_epoch = 8
    model_path = './model.ckpt'

    # 模型
    # model = MobileNetV1(classes=2, ch_in=3).to(device)  # 基于 MobileNetV1 简化的网络
    # model = SimpleModel().to(device)  # 简单搭建的神经网络
    model = res_model(2).to(device)  # ResNet18

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1, last_epoch=-1, verbose=True)

    best_acc = 0.0
    save_loss = []
    save_accs = []
    # 训练
    for epoch in range(num_epoch):
        if epoch >= 5:
            lr_scheduler.step()
        model.train()
        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader):
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            predict = model(imgs)
            loss = criterion(predict, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (predict.argmax(dim=-1) == labels).float().mean()

            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        print(f"[ Train | {epoch + 1:03d}/{num_epoch:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        model.eval()
        valid_loss = []
        valid_accs = []

        for batch in tqdm(test_loader):
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                predict = model(imgs)

            loss = criterion(predict, labels)

            acc = (predict.argmax(dim=-1) == labels.to(device)).float().mean()

            valid_loss.append(loss.item())
            valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        save_loss.append(valid_loss)
        save_accs.append(valid_acc)
        print(f"[ Valid | {epoch + 1:03d}/{num_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), model_path)
            print('saving model with acc {:.3f}'.format(best_acc))
