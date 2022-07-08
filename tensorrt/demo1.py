import onnx
import onnx_tensorrt.backend as backend
import numpy as np
import torch

model = onnx.load("/path/to/model.onnx")
#有其他GPU自行改造
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#后端加载模型就行了
engine = backend.prepare(model, device=device )
#你的输入 这里用随机数做测试
input_data = np.random.random(size=(32, 3, 224, 224)).astype(np.float32)
#engine.run就完事了
output_data = engine.run(input_data)[0]
