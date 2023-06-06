import caffe
from PIL import Image
import numpy as np
model_file = "best-simple.prototxt"
pretrained_file = "best-simple.caffemodel"
net = caffe.Net(model_file, pretrained_file, caffe.TEST)

image_path = 'ddd.png'
net.blobs['input'].reshape(1, 3, 224, 224)
# 以NCHW格式加载图像数据
transformer = caffe.io.Transformer({'input': net.blobs['input'].data.shape})
transformer.set_transpose('input', (2,0,1))
# blob = caffe.io.load_image(image_path)
# 加载图像并转换为 NumPy 数组
img = Image.open(image_path).convert('RGB')
img = np.array(img)

# 将 NumPy 数组传递给 Caffe 模型
# blob = caffe.io.array_to_blobproto(img)
# blob = caffe.io.load_image(image_path, False) 
transformed_image = transformer.preprocess('input', img)
# 输入数据
net.blobs['input'].data[...] = transformed_image
# 前向推理
output = net.forward()

print(output)