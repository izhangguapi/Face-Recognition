import os
import cv2
from PIL import Image
import numpy as np


# 获取脸部数据和标签
def getImageAndLabels(path):
    print('开始获取图像和标签')
    facesSamples = []
    ids = []
    names = []
    # 获取图片路径
    # imagePaths = []
    # for root,dirs,files in os.walk("./data"):
    #     for file in files:
    #         path = os.path.join(root,file)
    #         imagePaths.append(path)
    imagePaths = [
        os.path.join(path, f) for f in os.listdir(path)
        if not f.endswith('.DS_Store')
    ]
    print('数据排列：', imagePaths)
    # 检测人脸
    face_detector = cv2.CascadeClassifier(
        'haarcascades/haarcascade_frontalface_alt2.xml')
    # 遍历列表中的图片
    for imagePath in imagePaths:
        print('正在读取图片：', imagePath)
        # 打开图片,黑白化
        PIL_img = Image.open(imagePath).convert('L')
        # 将图像转换为数组，以黑白深浅
        img_numpy = np.array(PIL_img, 'uint8')
        # 获取图片人脸特征
        faces = face_detector.detectMultiScale(img_numpy)

        # 获取每张图片的id和姓名
        id = int(os.path.split(imagePath)[1].split('.')[0])
        image_name = os.path.split(imagePath)[1].split(".")[1]

        # 预防无面容照片
        for x, y, w, h in faces:
            ids.append(id)
            facesSamples.append(img_numpy[y:y + h, x:x + w])
            names.append(image_name)
    return facesSamples, ids


if __name__ == '__main__':
    # 图片路径
    path = './data/'
    # 获取图像数组和id
    faces, ids = getImageAndLabels(path)
    # 获取训练对象
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # recognizer.train(faces,names)#np.array(ids)
    recognizer.train(faces, np.array(ids))
    # 保存文件
    recognizer.write('trainer/trainer.yml')
    # save_to_file('names.txt',names)
    print('训练完成')
