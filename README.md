# 2023-NUEDC-E-OpenCV

基于 核桃派 ZeroW 的低成本 高性能 低延迟 OpenCV 2023电赛E题 图像检测方案开源

视频演示: 

本项目仅包含图像识别检测部分, 各项功能基本都用面向对象的方式封装好了, 控制部分可基于与本项目进一步开发

对 OpenCV 感兴趣想入门的可以看看核桃派写的入门教程, 他们家产品文档的阅读体验都很不错

https://www.walnutpi.com/docs/category/opencv 

## 功能特性

本项目包含下面的功能

- 识别 1mm 细线四边形**四个顶点**和**中心点**的坐标, 且任意摆放产生各种透视形变都可以识别
- 识别靶纸**黑框**的中心线围成的四边形**四个顶点**坐标 (近似)
- 按需要平均分割线段, 返回运动路径坐标点
- 识别 **红光点** & **绿光点** 的坐标
- 一套图传系统, 可在浏览器显示检测画面

识别细线大框:
![image](https://github.com/DBinK/2023-NUEDC-E-OpenCV/assets/21201676/e0abc2b8-3e58-44b9-8bb0-96b43ea46e44)

识别靶纸:
![image](https://github.com/DBinK/2023-NUEDC-E-OpenCV/assets/21201676/0d16f1fb-72e7-4a44-b843-a587f98745f2)

透视形变:
![image](https://github.com/DBinK/2023-NUEDC-E-OpenCV/assets/21201676/d6f5290c-2d6d-48ab-a2a9-37bfc169910c)

---

相比电赛中常见的 OpenMV 方案, 我们方案比较突出的功能特性有:

- **低成本**
    - 任意百元级 Linux 开发板就能跑, 相比 OpenMV 类方案便宜太多
    - 本项目使用的是一款国产派: 核桃派 ZeroW, 搭载全志 618 芯片, 1G RAM , 这样的硬件配置已经完全够用。 这款核桃派在官方淘宝店买全新只需要 **109** 元, 相比之下, 最便宜的 OpenMV 开发板如 K210 都要 **250** 元左右
- **高性能**
    - 检测识别延迟平均在 0.03 s 左右, 每秒能跑个40~60帧, 相比 OpenMV 快非常多!
    - 图传显示延迟平均在 0.24 s 左右, 图传只是方便调试, 关闭后检测延迟更低
- **兼容性** & **可扩展性**
    - 可使用**任意** USB/IP **摄像头**, 清晰度取决于你的摄像头素质
    - 对摄像头**画质要求低**, 480P 甚至更低的画质就可以完成比赛题目大部分要求
    - **无线**/**有线**皆可的 数据传输 和 开发调试 工作流
    - 写了一套基于 HTTP/TCP 协议的图传方案,    可以把 检测画面 **推流**到局域网/互联网

## 解题思路

根据题目要求, 我们需要识别 600mm^2 的白板上 1mm 细线围成的 500mm^2 四边形 **四个顶点**和**中心点**的坐标, 识别靶纸**黑框**的一些位置信息, 然后控制红绿激光点在这些坐标之间移动, 所以我们还需要识别 **红光点** & **绿光点** 在白板上的坐标, 以实现对光点的闭环控制

对于白板上 1mm 的细线, 如果想要直接通过它来识别四边形, 即使距离只有 1m , 依然对摄像头要求极高, 常规摄像头采到的画面里, 通常不是一条条连续的线。因此, 我们决定转变思路, 白板和大框的大小都是固定的, 既然细线识别不到, 但白板边框 (下文简称**外框**) 的特征还是很明显的, 我们可以通过识别白板边框围成的四边形, 缩小一圈, 得到缩小后的四边形坐标, 即为1mm 细线四边形四个顶点的坐标 (下文简称**内框**)。与此同时, 我们也可以对靶纸外边框也可以做一样的操作, 来得到**黑框**的中心线围成的四边形**四个顶点**坐标

至此, 我们只用一个算法, 完成了两个需要的四边形顶点坐标识别!

再将识别到的顶点, 顺时针排个序, 两两之间连成线段, 切一些平分点, 就可以作为运动路径给控制部分读取了。对于中点, 直接用顶点连成对角线, 取其交点就是中点了

**理论存在, 实践开始!**

**找四边形**: 采集图像 → 灰度化 → 找边缘 → 找轮廓 → 从轮廓中筛选最大允许周长的四边形 → 得到顶点坐标

**缩小一圈**: → 从顶点坐标把白板拉直到铺满屏幕, 计算透视变换矩阵

→ 从缩小比例 (内框是 500mm / 600mm) 把坐标缩小一圈 , 得到小一圈的顶点坐标(临时)

→ 把小一圈的顶点坐标, 乘以透视变换矩阵的逆矩阵, 变换回原来的坐标系, 得到内框顶点坐标

至此, 我们最难的识别部分就完成了。上述算法在 OpenCV 都有现成的函数, 可以尝试自己实现一次来上手 OpenCV, 这一部分的源码都放在`quad_detector.py` 中

后面红绿点的识别大家都大同小异, 都是通过匹配不同阈值实现的, 此处便不再赘述, 具体实现看`point_detector.py` 

本项目是写打电赛校内赛练手写的, 还没上过真的大赛场, 也是本人写的第一个 OpenCV 项目, 还有很多考虑不周的地方, 欢迎大家批评指正

## 项目架构

- `stream.py` 我写的推流程序, 可以本地 (使用 `cv2.imshow` ) 或者远程 (使用 Flask 起一个 http 服务器) 显示检测画面, 二次开发可以把**控制程序**写在里面
- `point_detector.py` **红光点** & **绿光点** 检测器类
- `quad_detector.py` **四边形** 检测器类, 包括靶纸检测
- `/img` 存放供测试的图片
- `/output` 测试输出
- `/templates` 存放 Flask 服务器模版 HTML 文件

## 快速开始

### 环境

理论上任意拥有完整 Python 环境的设备都可以运行, 本项目在核桃派的运行环境如下:

```bash
       _,met$$$$$gg.           pi@WalnutPi
    ,g$$$$$$$$$$$$$$$P.        -----------
  ,g$$P"         """Y$$.".     OS: Debian GNU/Linux bookworm 12.5 aarch64
 ,$$P'               `$$$.     Host: walnutpi-1b
',$$P       ,ggs.     `$$b:    Kernel: 6.1.31
`d$$'     ,$P"'   .    $$$     Uptime: 37 mins
 $$P      d$'     ,    $$$P    Packages: 846 (dpkg)
 $$:      $.   -    ,d$$'      Shell: fish 3.6.0
 $$;      Y$b._   _,d$P'       Cursor: Adwaita
 Y$$.    `.`"Y$$$$P"'          Terminal: /dev/pts/2
 `$$b      "-.__               CPU: Cortex-A53 (4)
  `Y$$                         Memory: 304.19 MiB / 1.94 GiB (15%)
   `Y$$.                       Swap: Disabled
     `$$b.                     Disk (/): 3.32 GiB / 28.86 GiB (12%) - ext4
       `Y$$b.                  Local IP (wlan0): 192.168.100.171/24 *
          `"Y$b._              Locale: zh_CN.UTF-8
             `"""
- Python 3.11
- walnutpi 2.2.0 固件
```

### 运行

创建虚拟环境

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

安装依赖

```bash
pip3 install -r requirements.txt
```

运行推流程序

```bash
python3 ./stream.py 
```

测试各项检测模块

```powershell
python3 ./point_detector.py  # 测试红绿点检测
python3 ./quad_detector.py   # 测试四边形检测
python3 ./test.py            # 同时测试
```

## 使用指南

```python
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
    vertices, scale_vertices, intersection, points_list = quad.detect(img)
    img_detected = quad.draw(img)  # 绘制检测结果

    print("四边形数据:")
    print("外框顶点坐标:", quad.vertices)
    print("内框顶点坐标:", quad.scale_vertices)
    print("外框中点坐标:", quad.intersection)
    print("运动路径坐标:", quad.points_list)

    # 初始化点检测器
    point = point_detector.PointDetector()

    # 点检测结果 (红点坐标, 绿点坐标)
    point.detect(img,vertices)
    img_detected = point.draw(img_detected)  # 绘制检测结果

    print("点数据:")
    print("红点坐标:", point.red_point)
    print("绿点坐标:", point.green_point)

    cv2.imshow("img_src", img)
    cv2.imshow("img_detected", img_detected)
    cv2.imwrite("output/detected.jpg", img_detected)
    cv2.waitKey(0)
```

## 关于作者

bilibili [@DBin_K](https://space.bilibili.com/37968660)

Chanal [@DBinKBB](https://t.me/DBinKBB)
