import tempfile
from pathlib import Path
import numpy as np
import cv2 # opencv-python
from ultralytics import YOLO
import time
import datetime
import deep_sort.deep_sort.deep_sort as ds
import glob
import os
import tensorflow as tf
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image
from torchvision import transforms
from siamense_net import SiameseNet,VectorNet
def preprocess_image_for_mobilenet(image):
    # 调整图像尺寸为224x224
    resized_image = cv2.resize(image, (224, 224))
    
    # 将图像从BGR转换为RGB
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
   
    return rgb_image
def get_image_feature_vector(img,modelmatch):

    x = np.float32(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = modelmatch.predict(x)
    return features.flatten()
def sift_similarity(img1, img2):
    # 将图像转换为灰度图像
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 初始化SIFT检测器
    sift = cv2.SIFT_create()

    # 检测关键点和计算描述符
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    if des1 is not None:
        des1 = des1.astype(np.float32)
    if des2 is not None:
        des2 = des2.astype(np.float32)

    if des1 is None or des2 is None:
        print("No descriptors found in one of the images.")
        return 0
    if len(des1) < 2 or len(des2) < 2:
        print("Not enough descriptors for knnMatch.")
        return 0
    # 使用FLANN匹配器进行匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 仅保留优质匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 计算相似度评价值：用优质匹配的数量表示
    similarity_score = len(good_matches)
    return similarity_score
def pHashValue(src_img):
    # 如果图像是彩色的，将其转换为灰度图像
    if src_img.shape[2] == 3:
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    
    # 缩放尺寸到32x32
    img = cv2.resize(src_img, (32, 32), interpolation=cv2.INTER_LINEAR)
    
    # 转换为浮点型
    img = np.float32(img)
    
    # 进行DCT变换
    dst_img = cv2.dct(img)
    
    # 获取前8x8的DCT系数
    dct_values = dst_img[:8, :8].flatten()
    
    # 计算均值
    mean_val = np.mean(dct_values)
    
    # 计算哈希值
    hash_str = ''.join(['1' if val >= mean_val else '0' for val in dct_values])
    
    return hash_str
def hammingDist(str1, str2):
    # 检查字符串长度是否为64
    
    if len(str1) != 64 or len(str2) != 64:
        return -1

    # 计算汉明距离
    dist_value = sum(c1 != c2 for c1, c2 in zip(str1, str2))
    print(dist_value)
    return dist_value

def putTextWithBackground(img, text, origin, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1):
    """绘制带有背景的文本。

    :param img: 输入图像。
    :param text: 要绘制的文本。
    :param origin: 文本的左上角坐标。
    :param font: 字体类型。
    :param font_scale: 字体大小。
    :param text_color: 文本的颜色。
    :param bg_color: 背景的颜色。
    :param thickness: 文本的线条厚度。
    """
    # 计算文本的尺寸
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # 绘制背景矩形
    bottom_left = origin
    top_right = (origin[0] + text_width, origin[1] - text_height - 5)  # 减去5以留出一些边距
    cv2.rectangle(img, bottom_left, top_right, bg_color, -1)

    # 在矩形上绘制文本
    text_origin = (origin[0], origin[1] - 5)  # 从左上角的位置减去5来留出一些边距
    cv2.putText(img, text, text_origin, font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)
    
def extract_detections(results, detect_class):
    """
    从模型结果中提取和处理检测信息。
    - results: YoloV8模型预测结果，包含检测到的物体的位置、类别和置信度等信息。
    - detect_class: 需要提取的目标类别的索引。
    参考: https://docs.ultralytics.com/modes/predict/#working-with-results
    """
    
    # 初始化一个空的二维numpy数组，用于存放检测到的目标的位置信息
    # 如果视频中没有需要提取的目标类别，如果不初始化，会导致tracker报错
    detections = np.empty((0, 4)) 
    
    confarray = [] # 初始化一个空列表，用于存放检测到的目标的置信度。
    
    # 遍历检测结果
    # 参考：https://docs.ultralytics.com/modes/predict/#working-with-results
    for r in results:
        for box in r.boxes:
            # 如果检测到的目标类别与指定的目标类别相匹配，提取目标的位置信息和置信度
            if box.cls[0].int() == detect_class:
                x1, y1, x2, y2 = box.xywh[0].int().tolist() # 提取目标的位置信息，并从tensor转换为整数列表。
                conf = round(box.conf[0].item(), 2) # 提取目标的置信度，从tensor中取出浮点数结果，并四舍五入到小数点后两位。
                detections = np.vstack((detections, np.array([x1, y1, x2, y2]))) # 将目标的位置信息添加到detections数组中。
                confarray.append(conf) # 将目标的置信度添加到confarray列表中。
    return detections, confarray # 返回提取出的位置信息和置信度。

# 视频处理

def detect_and_track(transformsx,query_img:list,input_path: str, output_path: str, detect_class: int, model, tracker) -> Path:
    """
    处理视频，检测并跟踪目标。
    - input_path: 输入视频文件的路径。
    - output_path: 处理后视频保存的路径。
    - detect_class: 需要检测和跟踪的目标类别的索引。
    - model: 用于目标检测的模型。
    - tracker: 用于目标跟踪的模型。
    """
    modelmatch = MobileNet(weights='imagenet', include_top=False, pooling='avg')
    cap = cv2.VideoCapture(input_path)  # 使用OpenCV打开视频文件。
    if not cap.isOpened():  # 检查视频文件是否成功打开。
        print(f"Error opening video file {input_path}")
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) # 获取视频的分辨率（宽度和高度）。
    output_video_path = Path(output_path) / "output.avi" # 设置输出视频的保存路径。
        # 设置视频编码格式为XVID格式的avi文件
    # 如果需要使用h264编码或者需要保存为其他格式，可能需要下载openh264-1.8.0
    # 下载地址：https://github.com/cisco/openh264/releases/tag/v1.8.0
    # 下载完成后将dll文件放在当前文件夹内
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    ind=0# 记录帧数
    query_img_info=a = [[] for _ in range(len(query_img))]
    img_info=[]
    Video_info_List=[]
    img_info_list=[]
    for idx,img_name in enumerate(query_img):
            img=cv2.imread(img_name)
            #img = cv2.resize(img,(192, 128), interpolation=cv2.INTER_LINEAR)
            
            #img=transformsx(img)          
            #img = img.unsqueeze(0)
            img_info.append(img)
    
    # 对每一帧图片进行读取和处理
    while True:
        success, frame = cap.read() # 逐帧读取视频。
        
        # 如果读取失败（或者视频已处理完毕），则跳出循环。
        if not (success):
            break
        if ind%60!=0:
            ind+=1
            continue
        # 使用YoloV8模型对当前帧进行目标检测。
        
        else:
            frame=frame[int(frame.shape[0]/2):,:,:]
            results = model(frame, stream=True)
            
            # 从预测结果中提取检测信息。
            detections, confarray = extract_detections(results, detect_class)
            
            # 使用deepsort模型对检测到的目标进行跟踪。
            #resultsTracker = tracker.update(detections, confarray, frame)

            # 使用deepsort模型对检测到的目标进行跟踪。
            for idnp in range(detections.shape[0]):
                if(confarray[idnp]>0.3):
                    x1=detections[idnp,0]
                    y1=detections[idnp,1]
                    x2=detections[idnp,0]+detections[idnp,2]
                    y2=detections[idnp,1]+detections[idnp,3]
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])           
                    img2=frame[y1:y2,x1:x2,:]                    
                    #img2=transformsx(img2)
                    #img2 = img2.unsqueeze(0)
                    Video_info_List.append((ind,img2))  
            print(f"frameidx:{ind}")
        ind+=1

            
        
    add_info=deepcopy(query_img_info)

    for index_of_query,query_imgx in enumerate(img_info):
        distance=[]
        features1=get_image_feature_vector(preprocess_image_for_mobilenet(query_imgx),modelmatch)
        #features1=pHashValue(query_imgx)
        for detectedimg in Video_info_List:
            features2=get_image_feature_vector(preprocess_image_for_mobilenet(detectedimg[1]),modelmatch)
            #features2=pHashValue(detectedimg[1])
            similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
            #similarity=-hammingDist(features1,features2)
            print(f"Index_query:{index_of_query},frame_index:{detectedimg[0]},Similarity:{similarity}")
            distance.append((detectedimg[0],similarity))
        sorted_distance = sorted(distance, key=lambda x: x[1], reverse=True)
        unique_index_results = []
        seen_indices = set()
        for item in sorted_distance:
            index = item[0]
            if index not in seen_indices:
                unique_index_results.append(item)
                seen_indices.add(index)
            if len(unique_index_results) == 5:
                break
        img_info_list.append(unique_index_results)
    
    '''
    for  img_id,query_obj in enumerate(query_img_info):
        imgx=cv2.imread(query_img[img_id])
        imgx = cv2.resize(imgx,(192,128), interpolation=cv2.INTER_LINEAR)
        
        imgx=transformsx(imgx)
        imgx = imgx.unsqueeze(0)
        if (len(query_obj)-1)==0:
            for t in range(60-1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, query_obj[0]+t+1)
                success, frame = cap.read() # 逐帧读取视频。
                frame=frame[int(frame.shape[0]/2):,:,:]
                results = model(frame, stream=True)
                
            # 从预测结果中提取检测信息。
                detections, confarray = extract_detections(results, detect_class)
                
                
            # 使用deepsort模型对检测到的目标进行跟踪。
                for idnp in range(detections.shape[0]):
                    if(confarray[idnp]>0.3):
                        x1=detections[idnp,0]
                        y1=detections[idnp,1]
                        x2=detections[idnp,0]+detections[idnp,2]
                        y2=detections[idnp,1]+detections[idnp,3]
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])            
                        img2=frame[y1:y2,x1:x2,:]
                        img2 = cv2.resize(img2,(192,128), interpolation=cv2.INTER_LINEAR)
                        
                        
                        img2=transformsx(img2)
                        img2 = img2.unsqueeze(0)
                        output1,output2=Similaritynet(imgx,img2)                            
                        similarity = F.pairwise_distance(output1, output2)
                        if(similarity<0.002):
                            add_info[img_id].append(query_obj[0]+t+1)
                            break
        else:
            for k in range(len(query_obj)-1):
                if query_obj[k]==0:
                    for t in range(60-1):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, query_obj[k]+t+1)
                        success, frame = cap.read() # 逐帧读取视频。
                        frame=frame[int(frame.shape[0]/2):,:,:]
                        results = model(frame, stream=True)
                        
                    # 从预测结果中提取检测信息。
                        detections, confarray = extract_detections(results, detect_class)
                        
                        
                    # 使用deepsort模型对检测到的目标进行跟踪。
                        for idnp in range(detections.shape[0]):
                            if(confarray[idnp]>0.3):
                                x1=detections[idnp,0]
                                y1=detections[idnp,1]
                                x2=detections[idnp,0]+detections[idnp,2]
                                y2=detections[idnp,1]+detections[idnp,3]
                                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])            
                                img2=frame[y1:y2,x1:x2,:]
                                img2 = cv2.resize(img2,(192,128), interpolation=cv2.INTER_LINEAR)
                                
                                img2=transformsx(img2)   
                                img2 = img2.unsqueeze(0)                         
                                output1,output2=Similaritynet(imgx,img2)                            
                                similarity = F.pairwise_distance(output1, output2)
                                print("similarity:",similarity,",capindex:",query_obj[k]+t+1)
                                if(similarity<0.002):
                                    add_info[img_id].append(query_obj[k]+t+1)
                                    break
                else:
                    for t in range(60-1):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, query_obj[k]+t+1)
                        success, frame = cap.read() # 逐帧读取视频。
                        results = model(frame, stream=True)
                        
                    # 从预测结果中提取检测信息。
                        detections, confarray = extract_detections(results, detect_class)
                        
                        
                    # 使用deepsort模型对检测到的目标进行跟踪。
                        for idnp in range(detections.shape[0]):
                            if(confarray[idnp]>0.3):
                                x1=detections[idnp,0]
                                y1=detections[idnp,1]
                                x2=detections[idnp,0]+detections[idnp,2]
                                y2=detections[idnp,1]+detections[idnp,3]
                                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])            
                                img2=frame[y1:y2,x1:x2,:]
                                img2 = cv2.resize(img2,(192,128), interpolation=cv2.INTER_LINEAR)
                                img2 = cv2.resize(img2,(192,128), interpolation=cv2.INTER_LINEAR)
                                
                                img2=transformsx(img2)
                                img2 = img2.unsqueeze(0)
                                output1,output2=Similaritynet(imgx,img2)                            
                                similarity = F.pairwise_distance(output1, output2)
                                print("similarity:",similarity,",capindex:",query_obj[k]+t+1)
                                if(similarity<0.002):
                                    add_info[img_id].append(query_obj[k]+t+1)
                                    break
    '''
    cap.release()  # 释放视频文件。
    
    print(f'output dir is: {output_video_path}')
    return img_info_list,output_video_path

if __name__ == "__main__":
    # 指定输入视频的路径。
    start_time=time.time()
    ######
    input_path = "E:/Data_Mining/test/test.mp4"
    ######

    # 输出文件夹，默认为系统的临时文件夹路径
    output_path = tempfile.mkdtemp()  # 创建一个临时目录用于存放输出视频。
    transformt = transforms.Compose([transforms.ToPILImage(),transforms.Resize((192,128 )),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    # 加载yoloV8模型权重
    model = YOLO("yolov8n.pt")
    base_net=VectorNet()
    Similaritynet=SiameseNet(base_net,128)
    Similaritynet.load_state_dict(torch.load("E:/Data_Mining/Try1/yolotest/yolov8-deepsort-tracking/weights_epoch_31_batch_31.pth"))
    Similaritynet.eval()
    # 设置需要检测和跟踪的目标类别
    # yoloV8官方模型的第一个类别为'person'
    detect_class = 2
    print(f"detecting {model.names[detect_class]}") # model.names返回模型所支持的所有物体类别

    # 加载DeepSort模型
    tracker = ds.DeepSort("E:/Data_Mining/Try1/yolotest/yolov8-deepsort-tracking/deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    folder_path="E:/Data_Mining/test/query"

    query_inputs = glob.glob(folder_path+"/*")
    for i in query_inputs:
        print(i)
    query_info,__=detect_and_track(transformt,query_inputs,input_path, output_path, detect_class, model, tracker)
    print(query_info)
    '''
    for i in len(query_info):
        query_info[i]=np.array(query_info[i])
    np.set_printoptions(threshold=np.inf)
   
    for i in len(query_info):
        filename = f'output{i}.txt'
        np.savetxt(filename, query_info[i], fmt='%d') 
    for i in query_info:
        print("len:",len(i))
    '''
    end_time=time.time()
    print("time cost:", float(end_time - start_time) * 1000.0, "ms")
