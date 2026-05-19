import glob
import math
import os
import sys
from io import StringIO
from os.path import split
from pathlib import Path
from time import time

from scipy.stats import zscore
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from joblib import Parallel, delayed
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QLabel, QWidget, QPushButton, QGroupBox, \
    QToolButton, QStyle, QHBoxLayout, QProgressBar, QSlider, QTextEdit, QStatusBar, QVBoxLayout, QLineEdit, QComboBox, \
    QFileDialog, QCheckBox, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QGuiApplication, QColor, QIcon, QFont, QIntValidator

import facelab.Frame_Reading_Thread as Frame_Reading_Thread
import facelab.MainMenu as MainMenu
import facelab.class_for_ROI as class_for_ROI

import facelab.video_info as video_info


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceLab")
        self.setStyleSheet("QMainWndow {background:'black';}")
        # self.setGeometry(500, 100, 700, 800)
        app_icon = QIcon()
        icon_path = 'iconn.ico'
        app_icon.addFile(icon_path, QtCore.QSize(16, 16))
        app_icon.addFile(icon_path, QtCore.QSize(24, 24))
        app_icon.addFile(icon_path, QtCore.QSize(32, 32))
        app_icon.addFile(icon_path, QtCore.QSize(48, 48))
        app_icon.addFile(icon_path, QtCore.QSize(96, 96))
        app_icon.addFile(icon_path, QtCore.QSize(256, 256))
        QApplication.setWindowIcon(app_icon)
        QtCore.QCoreApplication.setApplicationName("Frame Extraction Pipeline")
        self.sizeObject = QGuiApplication.primaryScreen().availableGeometry()
        self.resize(self.sizeObject.width(), self.sizeObject.height())
        self.setAcceptDrops(True)
        self.setStyleSheet("QMainWindow {background:'black';}")

        ###################Draw ROIs##############
        self.add_roi_btn = QPushButton("ADD ROI")
        self.add_roi_btn.setFixedHeight(30)
        #####Combo box#############
        self.combo_box = QComboBox()
        self.combo_box.addItems(["Rectangular", "Circle","Mask out"])
        self.combo_box.setFixedHeight(30)
        # func = self.combo_box.currentText()


        #########ROI ImageView######
        self.roi_view = pg.GraphicsLayoutWidget()

        #######PLot Area##########################

        self.plot_window=pg.GraphicsLayoutWidget()

        ###############Image View#############
        self.image_window = pg.GraphicsLayoutWidget()

        #########Frame Number###########
        self.frame_num=QLineEdit()
        # validator = QIntValidator(0, 999999)
        # self.frame_num.setValidator(validator)
        self.frame_num.setReadOnly(True)
        self.frame_num.setFixedWidth(100)
        self.frame_num.setFixedHeight(30)
        self.frame_num.setFont(QFont("Arial", 8, QFont.Bold))

        self.videoframes=QLabel(self)
        self.videoframes.setFixedWidth(100)
        self.videoframes.setFixedHeight(100)
        self.videoframes.setFont(QFont("Arial", 8, QFont.Bold))
        # self.videoframes.setText("/1000000")

        ###########QEditLine##########################
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)

        ######ROI ##############
        self.roi_layout = QHBoxLayout()
        self.roi_layout.addWidget(self.add_roi_btn)
        self.roi_layout.addWidget(self.combo_box)

        #####Slider#################
        self.slider_frame = QSlider(self)
        self.slider_frame.setOrientation(QtCore.Qt.Horizontal)
        # self.slider_image.setInvertedAppearance(True)
        self.slider_frame.setStyleSheet("""
        QSlider::groove:horizontal {
            border: 1px solid #bbb;
            height: 6px;
            background: #ddd;
            margin: 0px;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background-color: blue;
            border: 10px solid blue;
            width: 10px;
            height: 10px;
            margin: -15px 10;
            border-radius: 0px;
        }
        """)


        self.slider_frame.setFixedWidth(int(0.8*self.sizeObject.width()))
        self.slider_frame.setMinimum(0)
        self.slider_frame.setDisabled(True)
        self.slider_frame.setValue(0)  # Initial value


        #####Load experiment info#########
        self.expe_info = QPushButton("Load Experiment info")
        self.exp_info_dispaly=QLabel(self)
        self.exp_info_dispaly.setFixedWidth(300)
        self.exp_info_dispaly.setWordWrap(True)

        #######Save Image button ####
        self.save_image = QPushButton("Save Image")
        self.save_image.setFixedHeight(30)

        #####Structural Similarity Index###########
        self.extract_ssim = QPushButton("Process")
        self.extract_ssim.setFixedHeight(30)

        #####keypoint model load#########
        # self.key_model = QPushButton("Select training model")
        # self.key_model_info = QLabel(self)
        # self.key_model_info.setFixedWidth(300)
        # self.key_model_info.setWordWrap(True)

        self.keycombo_layout=QHBoxLayout()
        self.key_combo_label=QLabel("Models: ")
        self.key_combo_label.setFont(QFont("Arial", 10))
        self.key_combo=QComboBox()
        self.key_combo.setFixedWidth(250)
        self.key_combo.setFont(QFont("Arial",10))
        self.keycombo_layout.addWidget(self.key_combo_label)
        self.keycombo_layout.addWidget(self.key_combo)



        self.blink=QCheckBox('Blink')
        self.blink.setFont(QFont("Arial", 10))
        self.blink.setEnabled(False)
        self.blink.setStyleSheet('color:black')

        self.ssim_extract=QCheckBox('SSIM')
        self.ssim_extract.setFont(QFont("Arial", 10))
        self.ssim_extract.setEnabled(False)
        self.ssim_extract.setStyleSheet('color:black')

        self.keypoint_check = QCheckBox('KeyPoint')
        self.keypoint_check.setFont(QFont("Arial", 10))
        self.keypoint_check.setEnabled(False)
        self.keypoint_check.setStyleSheet('color:black')

        # self.train_check = QCheckBox('Train')
        # self.train_check.setFont(QFont("Arial", 7, QFont.Bold))
        # self.train_check.setEnabled(False)
        # self.train_check.setStyleSheet('color:black')

        self.checkbox_layout=QHBoxLayout()
        self.checkbox_layout.addWidget(self.blink)
        self.checkbox_layout.addWidget(self.ssim_extract)
        self.checkbox_layout.addWidget(self.keypoint_check)
        # self.checkbox_layout.addWidget(self.train_check)

        #######Training button layout#####
        self.training_layout=QVBoxLayout()
        self.training_layout.addLayout(self.keycombo_layout)
        # self.training_layout.addWidget(self.key_model_info)

        #########PLay buttons###########
        self.play_btn = QPushButton("Play Button")
        self.play_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaPlay))
        self.pause_btn = QPushButton("Pause Button")
        self.pause_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaPause))

        self.play_layout = QHBoxLayout()
        self.play_layout.addWidget(self.play_btn)
        self.play_layout.addWidget(self.pause_btn)


        ####################

        self.roi_image_layout = QHBoxLayout()
        self.roi_image_layout.addWidget(self.image_window)
        self.roi_image_layout.addWidget(self.roi_view)

        #######Slider_image###########
        self.plot_slider_layout=QVBoxLayout()
        self.plot_slider_layout.layout().addWidget(self.plot_window)
        self.plot_slider_layout.layout().addWidget(self.slider_frame)


        #####################
        self.frame_layout = QHBoxLayout()
        self.frame_layout.addWidget(self.frame_num)
        self.frame_layout.addWidget(self.videoframes)


        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.scene_grid_layout = QGridLayout()
        self.central_widget.setLayout(self.scene_grid_layout)


        self.control_groupbox = QGroupBox("Control Area:")
        self.control_groupbox.setStyleSheet(
            """QGroupBox{border: 1px solid black;background-color:white;border-style:outset;border-radius:1px;color:black;padding:15px 0px}""")
        self.control_groupbox.setLayout(QGridLayout())
        self.control_groupbox.layout().setAlignment(Qt.AlignTop)

        self.log_groupbox = QGroupBox("Log Area:")
        # self.log_groupbox.setStyleSheet("""
        # QGroupBox {
        #     border: 2px solid black;
        #     border-radius: 5px;
        #     margin-top: 20px;
        # }
        # """)
        self.log_groupbox.setStyleSheet(
            """QGroupBox{border: 1px solid black;background-color:white;border-style:outset;border-radius:1px;color:black;padding:15px 0px}""")
        self.log_groupbox.setLayout(QGridLayout())
        self.log_groupbox.layout().setAlignment(Qt.AlignTop)

        self.frame_groupbox = QGroupBox("Frame:")
        self.frame_groupbox.setStyleSheet(
                        """QGroupBox{border: 1px solid black;background-color:white;border-style:outset;border-radius:1px;color:black;padding:15px 0px}""")
        self.frame_groupbox.setLayout(QGridLayout())
        self.frame_groupbox.layout().setAlignment(Qt.AlignTop)

        #####arranging the layouts and widgets in the main window
        self.scene_grid_layout.addLayout(self.roi_image_layout, 0, 2, 6, 7)
        self.scene_grid_layout.addLayout(self.plot_slider_layout, 6, 2, 8, 7)
        self.scene_grid_layout.addWidget(self.control_groupbox, 0, 0, 1, 2)
        self.scene_grid_layout.addWidget(self.log_groupbox, 1, 0, 2, 2)
        self.scene_grid_layout.addWidget(self.frame_groupbox, 3, 0, 10, 2)
        #################arranging the elements on the side of the main window
        self.control_groupbox.layout().addLayout(self.roi_layout, 0, 1, 1, 2)
        self.log_groupbox.layout().addWidget(self.log_display, 0, 1, 1, 2)
        self.frame_groupbox.layout().addLayout(self.frame_layout, 1, 1, 2, 2)
        self.frame_groupbox.layout().addLayout(self.play_layout, 2, 1, 3, 2)
        self.frame_groupbox.layout().addWidget(self.expe_info, 3, 1,4, 2)
        self.frame_groupbox.layout().addWidget(self.exp_info_dispaly, 6, 1, 7, 2)
        self.frame_groupbox.layout().addWidget(self.save_image, 10, 1, 11, 2)
        self.frame_groupbox.layout().addWidget(self.extract_ssim, 14, 1, 15, 2)
        self.frame_groupbox.layout().addLayout(self.checkbox_layout,17,1,18,2)
        self.frame_groupbox.layout().addLayout(self.training_layout, 29, 1, 30, 2)
        # self.frame_groupbox.layout().addWidget(self.key_model_info, 33, 1, 34, 2)


        self.image_viewbox = self.image_window.addViewBox(lockAspect=True,invertY=True)
        self.roi_viewbox = self.roi_view.addViewBox(lockAspect=True, invertY=True)
        self.image_item = pg.ImageItem()
        self.scatter_item=pg.ScatterPlotItem()
        self.image_viewbox.addItem(self.image_item)
        self.image_viewbox.addItem(self.scatter_item)


        self.roi_pg = pg.ImageItem()
        # self.roi_pg.setImage(roi_im[1:100,1:100,:])
        self.roi_viewbox.addItem(self.roi_pg)
        pg.setConfigOptions(imageAxisOrder='row-major')

        MainMenu.mainmenu(self)

        self.frame_num.textChanged.connect(self.on_frame_changed)
        self.play_btn.clicked.connect(self.start_timer)
        self.pause_btn.clicked.connect(self.stop_timer)
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_video)
        self.add_roi_btn.clicked.connect(self.add_roi)
        self.expe_info.clicked.connect(self.load_experiment_info)
        # self.key_model.clicked.connect(self.load_keypoint_model)
        self.slider_frame.valueChanged.connect(self.update_frame_slider)
        self.save_image.clicked.connect(self.image_saving)
        self.extract_ssim.clicked.connect(self.ssim_extraction)

        self.add_roi_btn.setDisabled(True)
        self.play_btn.setDisabled(True)
        self.pause_btn.setDisabled(True)

        self.svd_plot_window_plot=self.plot_window.addPlot(name="ROI traces",row=0,col=0,title="ROI traces")
        self.svd_plot_window_plot2 = self.plot_window.addPlot(name="Keypoint traces", row=1, col=0, title="Keypoint traces")
        # self.svd_plot_window_plot.hideAxis("bottom")
        self.svd_plot_window_plot.layout.setContentsMargins(0,0,0,0)

        # res = np.load(r"C:\LocalExpData\blinking2\2025-04-22_1\extracted_ssim.npy")
        # self.x = np.linspace(0, len(res), len(res),dtype=int)
        # self.svd_plot_window_plot.plot(res,pen=svdpen)
        # self.svd_plot_window_plot.setLimits(xMin=0, xMax=len(res))

        # Connect event
        self.svd_plot_window_plot.scene().sigMouseClicked.connect(self.mouse_clicked)
        self.svd_plot_window_plot.hideAxis("left")
        self.svd_plot_window_plot.setMouseEnabled(x=True, y=False)
        self.vline = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('white', width=2))
        self.color = np.maximum(0, np.minimum(255, [0, 255, 50] + np.random.randn(3) * 70))


        # self.svd_plot_window_plot2.plot(
        #     np.array([1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 9, 1, 2, 3, 4, 5, 6, 0, 7, 5, 3, 2, 6, 8, 0, 4]) / 9, pen=svdpen)
        # self.svd_plot_window_plot2.plot(
        #     np.array([1, 2, -3, 4, 5, -5, 6, 6, 7, 7, -8, 90, 1, 2, 3, 4, 5, 6, 0, -7, 5, 30, 2, 6, -8, 0, -4]) / 9, pen=svdpen)

        self.video_path=[]
        self.ROIs=[]
        self.nROIs=0
        self.image=[]
        self.pos_holder = []
        self.overall_frame=0
        self.deleted_roi=[]
        self.save_masked_image=[]
        self.score_arr=[]
        self.temp_x=None
        self.temp_y=None
        self.temp_like=None
        self.keypoint_model=None
        self.ssim_plot_loaded=False
        self.frame_proto_loaded=False
        self.keypoint_combo_item()
        self.show()


    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            url=event.mimeData().urls()[0]
            if url.isLocalFile() and url.toString().endswith(('.mp4','.avi','.mkv','.mov')):
               video_path=url.toLocalFile()
               MainMenu.load_video(self,drag=True,video=video_path)


    def mouse_clicked(self,event):
        if self.ssim_plot_loaded:
            if event.button() == QtCore.Qt.LeftButton:
                pos = event.scenePos()
                if self.svd_plot_window_plot.sceneBoundingRect().contains(pos):
                    mouse_point = self.svd_plot_window_plot.vb.mapSceneToView(pos)
                    x_click = mouse_point.x()
                    # Find index of closest x
                    idx = int(np.abs(self.x - x_click).argmin())
                    self.vline.setValue(self.x[idx])
                    video_info.get_frame_number(self, self.video_path[0], self.x[idx],tem_x=self.temp_x,tem_y=self.temp_y,tem_like=self.temp_like)

    def ssim_plot(self,ssim_file):
        self.svd_plot_window_plot.clear()
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('white', width=2))
        self.vline.setValue(0)
        self.svd_plot_window_plot.addItem(self.vline)
        res = np.load(ssim_file,allow_pickle=True).item()
        self.x = np.linspace(0, len(res["blink"][0]), len(res["blink"][0]),dtype=int)
        for ind_plt,val_plt in enumerate(res["blink"]):
            color =res['ROI'][ind_plt][3]
            self.svdpen = pg.mkPen(color, width=2)
            self.svd_plot_window_plot.plot(zscore(val_plt), pen=self.svdpen)
        self.svd_plot_window_plot.setRange(xRange=(0, len(res["blink"][0])))
        self.svd_plot_window_plot.setLimits(xMin=0, xMax=len(res["blink"][0]))
        self.svd_plot_window_plot.enableAutoRange('xy', True)
        self.svd_plot_window_plot.autoRange()  # This immediately applies it
        self.load_roi(res)

    def on_frame_changed(self):
        if self.frame_num.text()=="":
           self.frame_num.setText("0")

        else:
            self.current_frame = int(self.frame_num.text())
            if int(self.frame_num.text())>self.overall_frame:
                self.frame_num.setText(str(self.overall_frame))
            self.vline.setValue(self.frame_num.text())
            # self.log_display.append(self.frame_num.text())
            video_info.get_frame_number(self,self.video_path[0],int(self.frame_num.text()),tem_x=self.temp_x,tem_y=self.temp_y,tem_like=self.temp_like)
        self.slider_frame.setValue(int(self.frame_num.text()))
    def start_timer(self):
        self.slider_frame.setDisabled(True)
        self.timer.start(50)
    def stop_timer(self):
        self.slider_frame.setDisabled(False)
        self.timer.stop()

    def play_video(self):

        self.current_frame +=1
        self.slider_frame.setValue(self.current_frame)
        video_info.get_frame_number(self, self.video_path[0], int(self.current_frame),tem_x=self.temp_x,tem_y=self.temp_y,tem_like=self.temp_like)
        if self.current_frame==self.overall_frame:
            self.stop_timer()

    def load_roi(self,res):
        if self.ROIs:
            for roi in self.ROIs:
                self.image_viewbox.removeItem(roi.ROI)
        self.ROIs=[]
        self.nROIs=0
        self.pos_holder = []
        for roi in res["ROI"]:
            self.log_display.append("One ROI added")
            self.ROIs.append(class_for_ROI.class_for_roi(roi_pos=roi[0],roi_size=roi[1],roi_shape=roi[2],roi_color=roi[3],parent=self))
            self.nROIs += 1
            self.ROIs[-1].position(self)

    def add_roi(self):

        self.log_display.append("One ROI added")
        self.ROIs.append(class_for_ROI.class_for_roi(parent=self))
        self.nROIs +=1
        self.ROIs[-1].position(self)


    def load_keypoint_model(self):
        keypoint_model, _=QFileDialog.getOpenFileName(
            self,
            "Select keypoint model (.pt)",
            "C:\LocalExpData",
            "pt Files (*.pt);"
        )
        if keypoint_model:
            self.key_model_info.setText(keypoint_model)
            self.keypoint_model=keypoint_model

        # if len(self.keypoint_model)==0: self.keypoint_model=None

    def load_experiment_info(self):
        self.file_path, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Open Experiment info file",
            directory="C:\LocalExpData",
            filter="frame_proto, npy Files (*protfo*.npy);"
        )
        if self.file_path:
            try:
                self.log_display.append(f"{self.file_path} added")
                self.experiment_information=np.load(self.file_path,allow_pickle=True)
                self.experiment_design=self.experiment_information[0][0]
                self.experiment_frames = self.experiment_information[0][1]
                self.experiment_frame_count = self.experiment_information[0][2]
                self.frame_exclude = self.experiment_information[0][3]
                self.experiment_length = self.experiment_information[0][4]
                self.log_display.append(f"Experiment design: {self.experiment_design}")
                self.log_display.append(f"Event Frames: {self.experiment_frames}")
                self.log_display.append(f"Frame each experiment: {self.experiment_frame_count}")
                self.log_display.append(f"Frames to exclude: {self.frame_exclude}")
                self.log_display.append(f"Experiment Length: {self.experiment_length}")
                self.exp_info_dispaly.setText(self.file_path)
                self.frame_proto_loaded = True
            except:
                self.frame_proto_loaded = False
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Critical)
                msg.setText("Please select a correct file")
                msg.setWindowTitle("Error!")
                # msg.setStandardButtons(QMessageBox.StandaradButton.Ok)
                msg.exec_()


    def update_frame_slider(self):

        self.current_frame = self.slider_frame.value()
        self.frame_num.setText(str(self.slider_frame.value()))
        video_info.get_frame_number(self, self.video_path[0], self.slider_frame.value(),tem_x=self.temp_x,tem_y=self.temp_y,tem_like=self.temp_like)

    def image_saving(self):
        if self.ROIs and len(self.save_masked_image)>0:
            cv2.imwrite(os.path.join(split(self.video_path[0])[0],f"masked_{self.current_frame}.png"), self.save_masked_image)

    def log_dis_message(self,message):
        self.log_display.append(message)



    def ssim_extraction(self):
        # if self.train_check.isChecked():
        #     self.keypoint_check.setChecked(False)
        #     self.blink.setChecked(False)
        #     keypoint_data_path=Path(self.video_path[0]).parent
        #     train_model.initiate_training(keypoint_data_path,self.keypoint_model)

        blink_arr=[]
        if self.ROIs:
            self.ssim_arr = {}
            self.ssim_arr["blink"] = []
            self.ssim_arr["ROI"] = []
            self.ssim_arr["blink"].extend([[] for i in range(len(self.pos_holder)) if self.pos_holder[i][2] == 'Circle'])
            blink_arr = [i for i in self.pos_holder if i[2] == 'Circle']
        if self.keypoint_check.isChecked() or self.blink.isChecked():
            self.thread = Frame_Reading_Thread.FrameReaderThread(self.video_path[0], self.overall_frame,blink_arr,key_model=self.model_key_name[self.key_combo.currentIndex()],parent=self)
            self.thread.log_signal.connect(self.log_dis_message)
            self.thread.start()
            self.slider_frame.setDisabled(True)

    def keypoint_combo_item(self):
        self.key_combo.clear()
        model_dir = r"C:\LocalExpData\models"
        os.makedirs(model_dir, exist_ok=True)
        model_list0 = glob.glob(os.path.join(model_dir, '*.pt'))
        self.model_key_name = ["basic_model.pt"]
        self.model_key_name.extend([m_l for m_l in model_list0 if m_l != self.model_key_name[0]])
        combo_name = [Path(m_l).stem for m_l in self.model_key_name]
        self.key_combo.addItems(combo_name)


    def save_roi_info(self,rois):
        for roi in rois:
            self.ssim_arr["ROI"].append([[roi.ROI.pos().y(),roi.ROI.pos().x()], [roi.ROI.size().y(),roi.ROI.size().x()], roi.roi_shape,roi.color])


def parallerl_processing(video,rois,j,ref_image):
    if j%1000==0:
        print(j)
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, j)
    ret, frame0 = cap.read()
    if ret:
        img=cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
        # mask_image=np.zeros_like(img)
        for i in range(len(rois)):
            if rois[i][2]=="Mask out":
                img=img[rois[i][1][0]:rois[i][1][-1],rois[i][0][0]:rois[i][0][-1]]

        for i in range(len(rois)):
            if rois[i][2] != "Mask out":
                img[rois[i][1], rois[i][0]] = [0]

        score, dif = ssim(ref_image, img, full=True)

    return score





app = QApplication(sys.argv)
dialog = MainWindow()
dialog.showMaximized()
# dialog.show()
ret=app.exec_()
sys.exit(ret)


