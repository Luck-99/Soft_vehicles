# import os
# os.environ["CUDA_DEVICE_ORDER"] = "-1"
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from gui import *
import copy
from counter import CounterThread
from utils.sort import *
from model import *
from utils.utils import *
from utils.datasets import *

from config import *
from models.yolo import Model

class App(QMainWindow,Ui_mainWindow):
    #用来设置显示的主页面
    def __init__(self):
        super(App,self).__init__()
        self.setupUi(self)
        self.label_image_size = (self.label_image.geometry().width(),self.label_image.geometry().height())
        self.video = None
        self.exampleImage = None
        self.imgScale = None
        self.get_points_flag = 0
        self.countArea = []
        self.road_code = None
        self.time_code = None
        self.show_label = names   #这里用来定义想要显示的标签

        #button function 设置按钮功能
        self.pushButton_selectArea.clicked.connect(self.select_area)
        self.pushButton_openVideo.clicked.connect(self.open_video)
        self.pushButton_start.clicked.connect(self.start_count)
        self.pushButton_pause.clicked.connect(self.pause)
        self.label_image.mouseDoubleClickEvent = self.get_points


        self.pushButton_selectArea.setEnabled(False)
        self.pushButton_start.setEnabled(False)
        self.pushButton_pause.setEnabled(False)

        #some flags 设置一些标志信息
        self.running_flag = 0
        self.pause_flag = 0
        self.counter_thread_start_flag = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #这里设置跑的设备

        #设置数据配置路径，权重路径。模板定义路径
        data_config = "config/coco.yaml"
        weights_path = "weights/yolov5s.pt"
        model_def = "config/yolov5s.yaml"

        # counter Thread   计算车辆数量的线程
        self.counterThread = CounterThread()
        self.counterThread.sin_counterResult.connect(self.show_image_label)
        self.counterThread.sin_done.connect(self.done)
        self.counterThread.sin_counter_results.connect(self.update_counter_results)



    def open_video(self):
        openfile_name = QFileDialog.getOpenFileName(self,'打开视频','','Video files(*.avi , *.mp4)')  #设置选取视频的类型
        self.videoList = [openfile_name[0]]
        vid = cv2.VideoCapture(self.videoList[0])  #打开视频获取内容     0可以调用摄像头

        # self.videoWriter = cv2.VideoWriter(openfile_name[0].split("/")[-1], cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 10, (1920, 1080))
        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                self.exampleImage = frame
                self.show_image_label(frame)
                self.imgScale = np.array(frame.shape[:2]) / [self.label_image_size[1], self.label_image_size[0]]
                vid.release()
                break
        #设置按钮
        self.pushButton_selectArea.setEnabled(True)
        self.pushButton_start.setText("开始")
        self.pushButton_start.setEnabled(False)
        self.pushButton_pause.setText("暂停")
        self.pushButton_pause.setEnabled(False)

        #clear counting results
        KalmanBoxTracker.count = 0
        self.label_sum.setText("0")
        self.label_sum.repaint()


    def get_points(self, event):
        if self.get_points_flag:
            x = event.x()
            y = event.y()
            self.countArea.append([int(x*self.imgScale[1]),int(y*self.imgScale[0])])
            exampleImageWithArea = copy.deepcopy(self.exampleImage)
            for point in self.countArea:
                exampleImageWithArea[point[1]-10:point[1]+10,point[0]-10:point[0]+10] = (0,255,255) #设置区域颜色
            cv2.fillConvexPoly(exampleImageWithArea, np.array(self.countArea), (0,0,255))
            self.show_image_label(exampleImageWithArea)
        print(self.countArea)


    def select_area(self):
        #change Area needs update exampleImage
        if self.counter_thread_start_flag:
            ret, frame = self.videoCapture.read()
            if ret:
                self.exampleImage = frame
                self.show_image_label(frame)

        if not self.get_points_flag:
            self.pushButton_selectArea.setText("提交区域")
            self.get_points_flag = 1
            self.countArea = []
            self.pushButton_openVideo.setEnabled(False)
            self.pushButton_start.setEnabled(False)

        else:
            self.pushButton_selectArea.setText("选择区域")
            self.get_points_flag = 0
            exampleImage = copy.deepcopy(self.exampleImage)
            # painting area
            for i in range(len(self.countArea)):
                cv2.line(exampleImage, tuple(self.countArea[i]), tuple(self.countArea[(i + 1) % (len(self.countArea))]), (0, 0, 255), 2)
            self.show_image_label(exampleImage)

            #enable start button
            self.pushButton_openVideo.setEnabled(True)
            self.pushButton_start.setEnabled(True)


    def show_image_label(self, img_np):
        img_np = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
        img_np = cv2.resize(img_np, self.label_image_size)
        frame = QImage(img_np, self.label_image_size[0], self.label_image_size[1], QImage.Format_RGB888)  #构造好图像
        pix = QPixmap.fromImage(frame)

        self.label_image.setPixmap(pix)  #在框框里显示视频
        self.label_image.repaint()

    def start_count(self):
        if self.running_flag == 0:
            #clear count and display
            KalmanBoxTracker.count = 0
            for item in self.show_label:
                vars(self)[f"label_{item}"].setText('0')
            # clear result file
            # with open("results/results.txt", "a") as f:
            #     pass

            #start
            self.running_flag = 1
            self.pause_flag = 0
            self.pushButton_start.setText("停止")
            self.pushButton_openVideo.setEnabled(False)
            self.pushButton_selectArea.setEnabled(False)
            #emit new parameter to counter thread
            self.counterThread.sin_runningFlag.emit(self.running_flag)
            self.counterThread.sin_countArea.emit(self.countArea)
            self.counterThread.sin_videoList.emit(self.videoList)
            #start counter thread
            self.counterThread.start()

            self.pushButton_pause.setEnabled(True)


        elif self.running_flag == 1:  #push pause button
            #stop system
            self.running_flag = 0
            self.counterThread.sin_runningFlag.emit(self.running_flag)
            self.pushButton_openVideo.setEnabled(True)
            self.pushButton_selectArea.setEnabled(True)
            self.pushButton_pause.setEnabled(False)
            self.pushButton_start.setText("开始")



    def done(self,sin):
        if sin == 1:
            self.pushButton_openVideo.setEnabled(True)
            self.pushButton_start.setEnabled(False)
            self.pushButton_start.setText("开始")


    def update_counter_results(self,counter_results):  #右边显示的结果
        with open("results/results.txt", "a") as f:
            for i, result in enumerate(counter_results):
                label_var = vars(self)[f"label_{result[2]}"]
                label_var.setText(str(int(label_var.text())+1)) #类型
                label_var.repaint()
                label_sum_var = vars(self)[f"label_sum"]
                label_sum_var.setText(str(int(label_sum_var.text()) + 1)) #总数
                label_sum_var.repaint()
                f.writelines('  '.join(map(lambda x: str(x),result))+"\t"+str(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())))  #打印结果
                f.write(("\n"))
        # print("************************************************",len(counter_results))


    def pause(self):
        if self.pause_flag == 0:
            self.pause_flag = 1
            self.pushButton_pause.setText("继续")
            self.pushButton_start.setEnabled(False)
        else:
            self.pause_flag = 0
            self.pushButton_pause.setText("暂停")
            self.pushButton_start.setEnabled(True)

        self.counterThread.sin_pauseFlag.emit(self.pause_flag)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = App()
    myWin.show()
    sys.exit(app.exec_())
