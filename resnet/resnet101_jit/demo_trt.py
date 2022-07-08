import time

import torch

batch_size = 4
model = './checkpoint.pth'
dummy_input = torch.randn(batch_size, 3, 224, 224, device='cuda')
model = torch.load(model)
torch.onnx.export(model, dummy_input, "resnet101.onnx", verbose=False)

import torch
from torch.autograd import Variable
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import torch.nn as nn
import tqdm

batch_size = 4  # 这里的batch_size要与转换时的一致

trt_model_name = "./resnet101.trt"

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

# 操作缓存，进行运算， 这个函数是通用的
def infer(context, input_img, output_size, batch_size):
    # Convert input data to Float32,这个类型要转换，不严会有好多报错
    input_img = input_img.astype(np.float32)
    # Create output array to receive data
    output = np.empty(output_size, dtype=np.float32)

    # Allocate device memory
    d_input = cuda.mem_alloc(batch_size * input_img.nbytes)
    d_output = cuda.mem_alloc(batch_size * output.nbytes)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, input_img, stream)
    # Execute model
    context.execute_async(batch_size, bindings, stream.handle, None)
    # Transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)

    stream.synchronize()

    # Return predictions
    return output


# 执行测试函数
def do_test(context):
    for inputs, label in dataloaders['train']:  # 开始测试
        img = inputs.numpy()  # 这个数据要从torch.Tensor转换成numpy格式的
        label = Variable(label, volatile=True)
        output = infer(context, img, 102, 4)
    return label

def trt_infer():
    # 读取.trt文件
    def loadEngine2TensorRT(filepath):
        G_LOGGER = trt.Logger(trt.Logger.WARNING)
        # 反序列化引擎
        with open(filepath, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            return engine

    trt_model_name = './resnet101.trt'
    engine = loadEngine2TensorRT(trt_model_name)
    # 创建上下文
    context = engine.create_execution_context()
    start = time.time()
    loss = do_test(context)
    end = time.time() - start
    print(end)

if __name__ == '__main__':
    trt_infer()
