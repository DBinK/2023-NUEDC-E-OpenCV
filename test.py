import cv2
import quad_detector
import point_detector

if __name__ == '__main__':

    print("开始测试")
    
    img = cv2.imread("img/rgb.jpg")
    
    # 初始化四边形检测器
    quad = quad_detector.QuadDetector()

    quad.max_perimeter = 99999    # 四边形最大周长上限
    quad.min_perimeter = 1        # 四边形最小周长下限
    quad.scale         = 500/600  # 四边形缩放比例 内框 500/600, 靶纸 276/297
    quad.min_angle     = 30       # 四边形最小角度
    quad.line_seg_num  = 6        # 四边形线段分割数

    # 四边形检测结果
    quad.detect(img)
    img_detected = quad.draw(img)  # 绘制检测结果

    print("四边形数据:")
    print("外框顶点坐标:", quad.vertices)
    print("内框顶点坐标:", quad.scale_vertices)
    print("外框中点坐标:", quad.intersection)
    print("运动路径坐标:", quad.points_list)

    # 初始化点检测器
    point = point_detector.PointDetector()

    # 点检测结果 (红点坐标, 绿点坐标)
    point.detect(img, quad.vertices)  # 当传入 vertices 参数时, 会进行roi切割, 只检测四边形内的红绿点
    img_detected = point.draw(img_detected)  # 绘制检测结果

    print("点数据:")
    print("红点坐标:", point.red_point)
    print("绿点坐标:", point.green_point)

    cv2.imshow("img_src", img)
    cv2.imshow("img_detected", img_detected)
    cv2.imwrite("output/detected.jpg", img_detected)
    cv2.waitKey(0)