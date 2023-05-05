import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

# 加载训练数据集文件
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
names = {}


# 获取名字
def returnName():
    path = './data'

    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.DS_Store')]
    for i in image_paths:
        name = i.split('.')[2]  # name
        id = i.split('/')[2].split('.')[0]  # id
        # Windows用下面这两行
        # name = i.split('.')[2]  # name
        # id = i.split("\\")[1].split('.')[0]  # id
        names[id] = name
    # print(names)


# 解决中文是问号的问题
def draw_box_string(img, x, y, string):
    """
    img: imread读取的图片;
    x,y:字符起始绘制的位置;
    string: 显示的文字;
    return: img
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    # simhei.ttf 是字体，你如果没有字体，需要下载
    font = ImageFont.truetype("ttf/SimHei.ttf", 50, encoding="utf-8")
    draw.text((x, y - 60), string, (255, 0, 0), font=font)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img


# 准备识别的图片
def face_detect_demo(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度
    face_detector = cv2.CascadeClassifier(
        'haarcascades/haarcascade_frontalface_alt.xml')
    face = face_detector.detectMultiScale(gray)
    # face=face_detector.detectMultiScale(gray)
    for x, y, w, h in face:
        cv2.rectangle(img, (x, y), (x + w, y + h),
                      color=(0, 0, 255),
                      thickness=2)
        cv2.circle(img,
                   center=(x + w // 2, y + h // 2),
                   radius=w // 2,
                   color=(0, 255, 0),
                   thickness=1)
        # 人脸识别
        ids, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        if confidence < 50:
            if ids not in names:
                name = names[str(ids)]
        else:
            name = '不知道'
        # cv2.putText(img, name, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        img = draw_box_string(img, x, y, name)
    cv2.imshow('正在识别', img)


if __name__ == '__main__':
    # 获取名字
    returnName()

    # 图片识别
    img = cv2.imread('tested/1.jpg')
    while True:
        face_detect_demo(img)
        if cv2.waitKey():
            break

    # 摄像头识别
    # cap = cv2.VideoCapture(0)

    # 视频识别
    # cap = cv2.VideoCapture('tested/1.mp4')
    # while True:
    #     flag, frame = cap.read()
    #     if not flag:
    #         break
    #     face_detect_demo(frame)
    #     if ord('q') == cv2.waitKey(10):
    #         break
    # cap.release()  # 关闭摄像头，释放内存

    # 关闭窗口
    cv2.destroyAllWindows()
