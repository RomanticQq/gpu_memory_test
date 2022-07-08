import os
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from tqdm import tqdm
import fire
from apex import amp

data_dir = '../flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
                                 transforms.CenterCrop(224),  # 从中心开始裁剪
                                 transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
                                 transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
                                 transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
                                 # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
                                 transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
                                 ]),
    'valid': transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True) for x in
               ['train', 'valid']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.NLLLoss()



def initialize_model():
    model_ft = models.resnet101(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102),
                                nn.LogSoftmax(dim=1))
    return model_ft


def train_model():
    model = initialize_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    # 设置使用精度的类型
    # O0 单精度   O1 混合精度   O3 半精度
    model = model.to(device)
    # model = model.to(device)
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    filename = 'checkpoint.pth'
    num_epochs = 1
    use_amp = True

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in tqdm(dataloaders['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # outputs = model(inputs.half())
            loss = criterion(outputs, labels)
            if use_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
        torch.save(model, filename)


def test_model():
    path = 'checkpoint.pth'
    model = torch.load(path)
    model.eval()
    loss_num = 0
    for inputs, labels in tqdm(dataloaders['train']):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss_num = loss_num + loss.item()

def test_trace_model():
    path = 'checkpoint.pth'
    path_1 = 'checkpoint_jit.pt'
    # script_moudle = torch.jit.script(torch.load(path))
    x = torch.randn([1, 3,  224, 224])
    trace_moudle = torch.jit.trace(torch.load(path), x.to(device))
    torch.jit.save(trace_moudle, path_1)
    model = torch.jit.load(path_1)
    model.eval()
    loss_num = 0
    for inputs, labels in tqdm(dataloaders['train']):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss_num = loss_num + loss.item()

if __name__ == '__main__':
    train_model()
