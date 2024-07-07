import cv2
import detector

if __name__ == '__main__':

    print("开始测试")
    
    img = cv2.imread("img/rgb.jpg")
    
    # 初始化四边形检测器
    quad_detector = detector.QuadDetector()

    quad_detector.max_perimeter = 99999
    quad_detector.min_perimeter = 1
    quad_detector.scale         = 500/600
    quad_detector.min_angle     = 30
    quad_detector.line_seg_num  = 6

    # 四边形检测结果
    vertices, scale_vertices, intersection, points_list = quad_detector.detect(img)
    img_detected = quad_detector.draw(img)  # 绘制检测结果

    # 初始化点检测器
    point_detector = detector.PointDetector()

    # 点检测结果
    red_point, green_point = point_detector.detect(img,vertices)
    img_detected = point_detector.draw(img_detected)  # 绘制检测结果

    cv2.imshow("img_src", img)
    cv2.imshow("img_detected", img_detected)
    cv2.imwrite("output/detected.jpg", img_detected)
    cv2.waitKey(0)