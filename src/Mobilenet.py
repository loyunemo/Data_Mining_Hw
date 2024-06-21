'''
Author: loyunemo 3210100968@zju.edu.cn
Date: 2024-06-06 14:59:46
LastEditors: loyunemo 3210100968@zju.edu.cn
LastEditTime: 2024-06-20 01:50:20
FilePath: \Data_Mining\Try1\yolotest\yolov8-deepsort-tracking\Mobilenet.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 加载预训练的MobileNet模型，不包括顶部的全连接层
model = MobileNet(weights='imagenet', include_top=False, pooling='avg')

def get_image_feature_vector(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

# 获取两张图片的特征向量
#features1 = get_image_feature_vector('image/1.jpg')
#features2 = get_image_feature_vector('image/2.jpg')

# 计算余弦相似度
#similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
#print("Similarity:", similarity)

# Similarity: 0.99791807
