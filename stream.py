import platform
import threading
import cv2
import time
import numpy as np

from flask import Flask, render_template, Response
from loguru import logger

import quad_detector, point_detector

# 创建检测器对象
# quad_detector = quad_detector.QuadDetector(9000, 100, 500/600, 5, 8)
quad_detector  = quad_detector.QuadDetector(1000, 1, 276/297, 5, 8)
point_detector = point_detector.PointDetector()

class ThreadedCamera(object):
    """
    多线程摄像头类
    """
    def __init__(self, url=0, FPS=1/60):
        """
        :param url: 摄像头地址, 默认为0, 表示使用第0个摄像头, 可以换成文件地址/网址
        """
        self.frame   = None
        self.capture = cv2.VideoCapture(url)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 5)  # 设置最大缓冲区大小

        self.FPS    = FPS  # 设置[检测]程序的采样速率,单位为秒, 默认为60帧每秒
        self.FPS_MS = int(self.FPS * 1000)
    
        self.detection_times = [] # 检测延迟

        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (status, frame) = self.capture.read()
                if status:
                    self.frame = frame.copy()

            time.sleep(self.FPS)

    def process_frame(self, frame):
        """
        :param frame: 输入帧
        :return: 处理后的帧
        """
        start_time = time.time()
    
        try:
            # 获取四边形的 顶点坐标, 中心点坐标, 路径坐标点
            vertices, scale_vertices, intersection, points_list = quad_detector.detect(frame)
            frame_drawed  = quad_detector.draw()   

        except Exception as e:
            logger.error(f"未识别到矩形: {e}")
            frame_drawed = frame 

        try:
            # 获取 红点 和 绿点 的坐标
            red_point, green_point = point_detector.detect(frame) #, quad_detector.vertices)
            frame_drawed = point_detector.draw(frame_drawed)     
        
        except Exception as e:
            logger.error(f"未识别到红绿点: {e}")
            frame_drawed = frame

        # 可以把控制代码放在这里, 此时控制频率和数据采样频率同步
        # 
        # 也可以另外再开一个线程, 只在需要时读取数据进行控制, 减小性能开销

        end_time = time.time()
        detection_time = end_time - start_time
        self.detection_times.append(detection_time)

        logger.info(f"检测延迟: {detection_time}")

        return frame_drawed

    def show_frame(self):  
        # 本地调试显示用
        cv2.namedWindow('Original MJPEG Stream', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Processed Stream', cv2.WINDOW_NORMAL)

        while True:

            frame = self.frame

            if frame is not None:
                try:
                    processed_frame = self.process_frame(frame)
                    if processed_frame is not None:
                        cv2.imshow('Processed Stream', processed_frame)
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue

            cv2.imshow('Original MJPEG Stream', frame)
            key = cv2.waitKey(self.FPS_MS)
            if key == 27:  # ESC键退出
                break

        cv2.destroyAllWindows()


# Flask 服务器相关代码

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global url
    camera = ThreadedCamera(url)

    return Response(generate_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')
def generate_frames(camera):
    """
    生成发送Flask服务器的视频帧
    """
    while True:
        frame = camera.frame

        if frame is not None:
            try:
                processed_frame = camera.process_frame(frame)
                if processed_frame is not None:
                    _, jpeg_buffer = cv2.imencode('.jpg', processed_frame)
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + jpeg_buffer.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue

# 主函数
if __name__ == '__main__':

    # 更改为数字0使用第0个硬件摄像头, 也可以使用视频文件地址或者视频流网址
    url = 'http://192.168.1.207:8080/video/mjpeg'  

    # 如果是Linux系统, 则使用 Flask 服务器远程调试
    if platform.system() != 'Linux': 
        app.run(host='0.0.0.0', debug=True)

    # 如果是Windows系统, 则使用本地摄像头调试
    else:
        camera = ThreadedCamera(url)
        camera.show_frame()
        
