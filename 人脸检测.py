# 导入opencv的模块
import cv2
import numpy as np

np.set_printoptions(threshold=np.inf)


# 定义人脸检测的函数
def face_detect(img):
    # 将图片转换为灰度图
    gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gary)
    # print('ndarray的维度: ', gary.ndim)
    # print('ndarray的形状: ', gary.shape)
    # print('ndarray的元素数量: ', gary.size)
    face_detector = cv2.CascadeClassifier(
        "haarcascades/haarcascade_frontalface_alt.xml")

    faces = face_detector.detectMultiScale(gary)
    # print(type(gary))
    # 在窗口中绘制矩形框住人脸
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow("结果", img)


# 打开摄像头
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('tested/1.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        print("摄像头打开失败")
        break
    face_detect(frame)
    if cv2.waitKey(1) == ord("q"):
        break

# img = cv2.imread('tested/5.jpg')
# while True:
#     face_detect(img)
#     if ord('q') == cv2.waitKey(1):
#         break
# 关闭摄像头
cap.release()
# 关闭窗口
cv2.destroyAllWindows()
