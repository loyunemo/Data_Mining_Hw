#%%
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
#%%
# 自定义数据集类
class SiameseDataset(Dataset):
    def __init__(self, csv_file, fold_path,transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.fold_path=fold_path
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img1_path = self.data_frame.iloc[idx, 1]
        img2_path = self.data_frame.iloc[idx, 2]
        similarity = self.data_frame.iloc[idx, 3]

        img1 = Image.open(fold_path+img1_path).convert("RGB")
        img2 = Image.open(fold_path+img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.tensor([similarity], dtype=torch.float32)

# 图像转换
transform = transforms.Compose([
    transforms.Resize((192,128 )),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 读取CSV文件
fold_path='E:/Data_Mining/Dataset/VehicleID/data/2016-CVPR-VehicleReId-dataset/'
csv_file = fold_path+'ground_truth_crowdsourced_avg_values.csv'
dataset = SiameseDataset(csv_file=csv_file,fold_path=fold_path, transform=transform)

# 将数据集划分为训练集和验证集
train_data, val_data = train_test_split(dataset, test_size=0.2, )

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
#%%
def make_layers(in_channels, out_channels, blocks, downsample):
    layers = []
    for _ in range(blocks):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels 
        if downsample:
            layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)
# 定义孪生神经网络
class VectorNet(nn.Module):
    def __init__(self, num_classes=128):
        super(VectorNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        self.layer1 = make_layers(64, 64, 2, False)
        self.layer2 = make_layers(64, 128, 2, True)
        self.layer3 = make_layers(128, 256, 2, True)
        self.layer4 = make_layers(256, 512, 2, True)
        self.avgpool = nn.AvgPool2d((6, 4), 1)


        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):

        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SiameseNet(nn.Module):
    def __init__(self, base_net,num_classes):
        super(SiameseNet, self).__init__()
        self.base_net = base_net
        self.Judge= nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, 1),
        )
    def forward_once(self, x):
        return self.base_net(x)

    def forward(self,input1,input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        return output1,output2



# 创建孪生神经网络
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./runs/fashion_mnist_experiment_1")
 
# 将训练数据图像写入TensorBoard
dataiter = iter(train_loader)
image1=torch.rand([32,3,192,128])
image2=image1


base_net = VectorNet(num_classes=128)
siamese_net = SiameseNet(base_net,128)
writer.add_graph(siamese_net, (image1,image2))
siamese_net.eval()
output1,output2 = siamese_net(image1,image2)
euclidean_distance = F.pairwise_distance(output1,output2)
print(euclidean_distance)
#%%
'''#%%
# 定义损失函数和优化器
writer = SummaryWriter('runs/siamese_experiment')
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, img1, img2, label):
        euclidean_distance = F.pairwise_distance(img1, img2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
criterion = ContrastiveLoss(margin=1.0)  # 对比损失
optimizer = torch.optim.Adam(siamese_net.parameters(), lr=0.001)

# 训练过程
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        img1, img2, labels = data
        optimizer.zero_grad()
        output1,output2 = model(img1, img2)
        loss = criterion(output1,output2, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        weight_filename = f'weights_epoch_{epoch+1}_batch_{batch_idx+1}.pth'
                # 打印训练情况
        print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # 记录到 TensorBoard
        writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)
    torch.save(model.state_dict(), weight_filename)
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            img1, img2, labels = data
            output1,output2 = model(img1, img2)
            loss = criterion(output1,output2, labels)
            val_loss += loss.item()
    print(f"Validation Loss: {val_loss / len(val_loader)}")

# 训练和验证
num_epochs = 100
for epoch in range(num_epochs):
    train(siamese_net, train_loader, criterion, optimizer, epoch)
    validate(siamese_net, val_loader, criterion)
writer.close()
'''
# %%
