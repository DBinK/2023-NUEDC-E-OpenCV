import cv2
import numpy as np
import sys
from loguru import logger


# 设置日志级别 
logger.remove()
# logger.add(sys.stderr, level="DEBUG")  
logger.add(sys.stderr, level="INFO") 

class PointDetector:
    """
    @description: 点检测类
    """
    def __init__(self):
        self.img = None
        self.roi_vertices = None

        self.red_point   = None
        self.green_point = None

    def roi_cut(self, img, roi_vertices):
        """
        @description: ROI 区域裁剪
        @param img: 输入图像
        @param roi_vertices: ROI 区域的四个顶点坐标
        @return: 裁剪后的图像
        """
        roi_vertices = np.array(roi_vertices, dtype=np.int32)
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [roi_vertices], (255, 255, 255))
        masked_image = cv2.bitwise_and(img, mask)

        return masked_image

    @staticmethod
    def find_point(image):
        """
        @description: 寻找红点绿点的坐标
        @param image: 输入图像
        @return: 红点坐标, 绿点坐标
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        def find_max_contours(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                # 找到最大的轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                # 找到最大轮廓的外接矩形
                x, y, w, h = cv2.boundingRect(largest_contour)  
                
                center_x = int(x + w / 2)  # 计算中心点 x 坐标
                center_y = int(y + h / 2)  # 计算中心点 y 坐标

                point = [x, y, w, h, center_x, center_y]
                # print(point)
                return point
            else:
                return [0,0,0,0,0,0]
            
        def find_red_point(hsv):
            # 红色范围
            lower = np.array([0, 100, 100])
            upper = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower, upper)

            lower = np.array([160, 100, 100])
            upper = np.array([179, 255, 255])
            mask2 = cv2.inRange(hsv, lower, upper)

            mask = mask1 | mask2

            return find_max_contours(mask)
        def find_green_point(hsv):
            # 绿色范围
            lower = np.array([40, 100, 100])
            upper = np.array([80, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)

            return find_max_contours(mask)
        
        def find_yellow_point(hsv):
            # 黄色范围
            lower = np.array([26, 43, 46])
            upper = np.array([34, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            return find_max_contours(mask)
        
        def find_white_point(hsv):
            # 白色范围
            lower = np.array([0, 0, 200])
            upper = np.array([180, 30, 255])
            mask = cv2.inRange(hsv, lower, upper)
            return find_max_contours(mask)

        red_point = find_red_point(hsv)
        green_point = find_green_point(hsv)

        # 特殊情况处理
        if red_point[0] == 0 and green_point[0] == 0:
            yellow_point = find_yellow_point(hsv)
            red_point    = yellow_point
            green_point  = yellow_point
            logger.info("红绿光重叠, 找黄光")

            if yellow_point[0] == 0:
                white_point = find_white_point(hsv)
                red_point   = white_point
                green_point = white_point
                logger.info("黄光找不到, 找白光")

        logger.info(f"red_point: {red_point[4], red_point[5]} | green_point: {green_point[4], green_point[5]}")

        return red_point, green_point
    
    def detect(self, img, roi=[]):
        """
        @param img: 待检测图像
        @param roi: ROI 区域, 为空时不裁剪 
        """

        if len(roi) > 0:
            self.img = self.roi_cut(img.copy(), roi)
            # cv2.imshow("roi", self.img)
        else:
            self.img = img.copy()

        self.red_point, self.green_point = self.find_point(self.img)
        
        return self.red_point, self.green_point
    
    def draw(self, img=None):
        """
        @param img: 待绘制图像, 为空时使用内部图像
        """
        if img is None:
            img = self.img.copy()
        def draw_point(img, point, bgr = ( 0, 255, 255) , color = ' '):
            [x, y, w, h, center_x, center_y] = point
            # 在图像上绘制方框
            cv2.rectangle(img, (x, y), (x + w, y + h), bgr, 1)

            # 绘制坐标
            text = f"{color} point: ({center_x}, {center_y})"
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)

            return img
        
        img_drawed = draw_point(       img,   self.red_point, (0, 255, 255),   "red")
        img_drawed = draw_point(img_drawed, self.green_point, (0, 255, 255), "green")

        return img_drawed

if __name__ == '__main__':

    print("开始测试")
    
    img = cv2.imread("img/rgb.jpg")

    # 初始化点检测器
    point_detector = PointDetector()

    # 点检测结果
    red_point, green_point = point_detector.detect(img)
    img_detected = point_detector.draw(img)  # 绘制检测结果

    # 显示结果
    cv2.imshow("img_src", img)
    cv2.imshow("img_detected", img_detected)
    cv2.imwrite("output/detected.jpg", img_detected)
    cv2.waitKey(0)
    