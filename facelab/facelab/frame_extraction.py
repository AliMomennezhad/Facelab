import os
import shutil
from os.path import split
from pathlib import Path
from time import time

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QSize, QCoreApplication, Qt, QTimer, QPointF
from PyQt5.QtGui import QIcon, QGuiApplication, QFont
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QFileDialog, \
    QMessageBox, QGroupBox, QGridLayout, QCheckBox, QGraphicsProxyWidget, QGraphicsItem, QFrame, QStyle

from pyqtgraph.Qt import QtWidgets
import facelab.FrameExtractROI as FrameExtractROI
import facelab.Frame_Extraction_Preview as Frame_Extraction_Preview
import facelab.Progress_bar_extraction as Progress_bar_extraction


class Frame_Extraction(QtWidgets.QDialog):
    def __init__(self,parent=None):
        super().__init__(parent) #the window should inherit variables from the parent class otherwise it disappears immediately. for example we should show a picture or something. if we don't use anything we should pass a variable to __init__ of super
        self.setWindowFlags(
            Qt.Window |  # Basic window with title bar
            Qt.WindowMinimizeButtonHint |  # Enable minimize button
            Qt.WindowCloseButtonHint  # Enable close button
        )


        self.setGeometry(200, 100, 1500, 900)
        app_icon = QIcon()
        icon_path = r'keypoint_icon.png'
        app_icon.addFile(icon_path, QSize(16, 16))
        app_icon.addFile(icon_path, QSize(24, 24))
        app_icon.addFile(icon_path, QSize(32, 32))
        app_icon.addFile(icon_path, QSize(48, 48))
        app_icon.addFile(icon_path, QSize(96, 96))
        app_icon.addFile(icon_path, QSize(256, 256))
        QApplication.setWindowIcon(app_icon)
        QCoreApplication.setApplicationName("Frame Extraction GUI")
        centerPoint = QGuiApplication.primaryScreen().availableGeometry().center()
        qtRectangle = self.frameGeometry()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())
        pg.setConfigOptions(imageAxisOrder='row-major')
        mainlayout = QVBoxLayout()
        self.setLayout(mainlayout)
        #layout for containing Image and applied ROIs
        self.roi_image_layout=QHBoxLayout()

        #view box for showing frames
        self.imagewindow = pg.GraphicsLayoutWidget()
        self.video_view = self.imagewindow.addViewBox(lockAspect=True, invertY=True)
        self.image_item = pg.ImageItem()
        # self.check_item = QCheckBox('Excluded')
        # self.check_item.setStyleSheet("""QCheckBox{background-color:black;color:white}""")
        # self.check_item.setFont(QFont("Arial",14,QFont.Bold))
        self.video_view.addItem(self.image_item)
        # self.check_view=QGraphicsProxyWidget()
        # self.check_view.setWidget(self.check_item)
        # self.check_view.setParentItem(self.video_view)
        # self.check_view.setZValue(1000)

        # self.video_view.addItem(self.check_view,ignoreBounds=True)
        # self.video_view.sigResized.connect(self.check_box_position)
        # self.check_view.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        # self.check_box_position()
        #view box for showing frames after applying ROIs
        self.roiwindow = pg.GraphicsLayoutWidget()
        self.roi_view=self.roiwindow.addViewBox(lockAspect=True, invertY=True)
        self.roi_item=pg.ImageItem()

        self.roi_view.addItem(self.roi_item)

        self.roi_image_layout.addWidget(self.imagewindow)
        self.roi_image_layout.addWidget(self.roiwindow)

        #Here we make a frame and apply it to a layout. Then add the widgets to the layout. At the end we should add the frame to the main layout.
        self.grouping_top=QFrame()
        self.grouping_top.setFrameShape(QFrame.Box)
        self.grouping_top.setLineWidth(2)
        self.grouping_top.setStyleSheet("""QFrame{border:2px solid black; border-radius:5px}""")
        self.grouping_top_layout=QHBoxLayout(self.grouping_top)
        self.grouping_top_layout.setContentsMargins(10,10,10,10)


        self.bottom_frame=QFrame()
        self.bottom_frame.setFrameShape(QFrame.Box)
        self.bottom_frame.setLineWidth(2)
        self.bottom_frame.setStyleSheet("""QFrame{border:2px solid black; border-radius:5px}""")
        self.grouping_bottom_layout=QHBoxLayout(self.bottom_frame)
        self.grouping_bottom_layout.setContentsMargins(10,10,10,10)

        self.button_layout=QHBoxLayout()
        self.save_layout = QHBoxLayout()
        self.roi_layout = QHBoxLayout()

        self.next_button=QPushButton("Next")
        self.next_button.setFont(QFont("Arial",12,QFont.Bold))
        self.next_button.setFixedHeight(50)
        self.previous_button = QPushButton("Previous")
        self.previous_button.setFont(QFont("Arial",12,QFont.Bold))
        self.previous_button.setFixedHeight(50)
        self.roitype_button = QLabel("ROI Type: ")
        self.roitype_button.setFont(QFont("Arial",12,QFont.Bold))
        self.roitype_button.setFixedHeight(50)
        self.roitype_button.setFixedWidth(150)
        self.addroi_button = QPushButton("Add ROI")
        self.addroi_button.setFont(QFont("Arial",12,QFont.Bold))
        self.addroi_button.setFixedHeight(50)
        self.addroi_button.setFixedWidth(200)
        self.loadroi_button = QPushButton("Load ROI")
        self.loadroi_button.setFont(QFont("Arial",12,QFont.Bold))
        self.loadroi_button.setFixedHeight(50)
        self.loadroi_button.setFixedWidth(200)
        self.save_button = QPushButton("Save ROIs")
        self.save_button.setFont(QFont("Arial",12,QFont.Bold))
        self.save_button.setFixedHeight(50)
        self.save_button.setFixedWidth(150)

        self.extract_frames=QPushButton("Extract Frames")
        self.extract_frames.setFont(QFont("Arial",12,QFont.Bold))
        self.extract_frames.setFixedHeight(50)
        self.extract_frames.setFixedWidth(200)

        self.preview_btn=QPushButton("Preview")
        self.preview_btn.setFont(QFont("Arial",12,QFont.Bold))
        self.preview_btn.setFixedHeight(50)
        self.preview_btn.setFixedWidth(200)

        self.fr_combo_box = QComboBox()
        self.fr_combo_box.setFont(QFont("Arial",12,QFont.Bold))
        self.fr_combo_box.addItems(["Rectangular", "Circle", "Mask out"])
        self.fr_combo_box.setFixedHeight(50)
        self.fr_combo_box.setFixedWidth(200)
        # self.fr_combo_box.setStyleSheet("""QComboBox{background-color:#D5FFFF;}""")

        self.fr_extraction_mode = QComboBox()
        self.fr_extraction_mode.setFont(QFont("Arial", 12, QFont.Bold))
        self.fr_extraction_mode.addItems(["Individual", "All"])
        self.fr_extraction_mode.setFixedHeight(50)
        self.fr_extraction_mode.setFixedWidth(200)

        self.ext_wind_label = QLabel("Extraction window: ")
        self.ext_wind_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.ext_wind_label.setFixedHeight(50)
        self.ext_wind_label.setFixedWidth(200)
        self.fr_extraction_window = QComboBox()
        self.fr_extraction_window.setFont(QFont("Arial", 12, QFont.Bold))
        self.fr_extraction_window.addItems(["[7,5]","[7,7]","[7,0]","[6,0]","[3,0]"])
        self.fr_extraction_window.setFixedHeight(50)
        self.fr_extraction_window.setFixedWidth(200)

        self.button_layout.addWidget(self.previous_button)
        self.button_layout.addWidget(self.next_button)
        # self.button_layout.addWidget(self.addroi_button)
        # self.button_layout.addWidget(self.fr_combo_box)

        self.label_layout=QHBoxLayout()
        self.infor_label=QLabel("Information: ")
        self.infor_label.setFont(QFont("Arial",12,QFont.Bold))
        self.infor_label.setFixedHeight(50)
        self.frame_label = QLabel(self)
        self.frame_label.setFixedWidth(1000)
        self.frame_label.setFixedHeight(50)
        self.frame_label.setFont(QFont("Arial", 15, QFont.Bold))
        self.frame_label.setAlignment(Qt.AlignLeft)
        self.frame_label.setStyleSheet("""QLabel{background-color:green;}""")

        self.label_layout.addWidget(self.infor_label)
        self.label_layout.addWidget(self.frame_label)
        # self.save_layout.addWidget(self.loadroi_button)
        # self.save_layout.addWidget(self.extract_frames)
        # self.save_layout.addWidget(self.fr_extraction_mode)


        self.grouping_top_layout.addWidget(self.loadroi_button)
        self.grouping_top_layout.addWidget(self.addroi_button)
        self.grouping_top_layout.addWidget(self.roitype_button)
        self.grouping_top_layout.addWidget(self.fr_combo_box)
        self.grouping_top_layout.addWidget(self.save_button)

        self.grouping_bottom_layout.addWidget(self.preview_btn)
        self.grouping_bottom_layout.addWidget(self.extract_frames)
        self.grouping_bottom_layout.addWidget(self.fr_extraction_mode)
        self.grouping_bottom_layout.addWidget(self.ext_wind_label)
        self.grouping_bottom_layout.addWidget(self.fr_extraction_window)
        # self.save_layout.setAlignment(Qt.AlignCenter)
        mainlayout.addWidget(self.grouping_top)
        # mainlayout.addLayout(self.grouping_top_layout)
        mainlayout.addLayout(self.roi_image_layout)
        mainlayout.addLayout(self.button_layout)
        mainlayout.addLayout(self.label_layout)
        mainlayout.addWidget(self.bottom_frame)

        # self.video_path = parent.video_path[0]
        # cap = cv2.VideoCapture(self.video_path)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # ret, frame0 = cap.read()
        # image = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        self.video_path=parent.video_path[0]
        self.setWindowTitle(f"Frame Extraction- {self.video_path}")
        self.next_timer = QTimer()
        self.prev_timer = QTimer()
        self.next_button.clicked.connect(self.next_frame)
        self.next_timer.timeout.connect(self.next_frame)
        self.next_button.pressed.connect(lambda: self.start_slide_show("next"))
        self.next_button.released.connect(lambda: self.stop_slide_show("next"))
        self.previous_button.clicked.connect(self.previous_frame)
        self.prev_timer.timeout.connect(self.previous_frame)
        self.previous_button.pressed.connect(lambda: self.start_slide_show("prev"))
        self.previous_button.released.connect(lambda: self.stop_slide_show("prev"))
        self.preview_btn.clicked.connect(self.preview_function)
        self.save_button.clicked.connect(self.save_function)
        self.loadroi_button.clicked.connect(self.load_roi_Function)
        self.addroi_button.clicked.connect(self.addroi_function)
        self.extract_frames.clicked.connect(self.extract_frames_function)
        self.cap=None
        self.cap_extract=None
        self.fr_image = []
        self.parent = parent
        self.fr_ROIs=[]
        self.fr_nROIs=0
        self.fr_progress_value=0
        self.fr_progress_overall=0
        self.fr_deleted_roi = []
        self.fr_pos_holder = []
        self.frame_dictionary_meta={}
        self.extraction_window=[7,5]
        self.capturing_fs=65
        self.load_frame_info()
        self.load_video()
        # frame_extraction_menu.frame_extraction_menu(self)
        self.show()

    def load_video(self):
        if self.cap is not None:
            self.cap.release()
        self.cap=cv2.VideoCapture(self.video_path)
        self.counter = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.event_type[self.counter][0])

        ret,frame=self.cap.read()
        self.fr_image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        self.fr_image=cv2.resize(self.fr_image,(256,256))
        self.image_item.setImage(self.fr_image)
        if self.event_type[self.counter][1]=='reward':
            self.frame_label.setStyleSheet("""QLabel{background-color:blue;}""")
        if self.event_type[self.counter][1]=='airpuff':
            self.frame_label.setStyleSheet("""QLabel{background-color:red;}""")
        if self.event_type[self.counter][1]=='neutral':
            self.frame_label.setStyleSheet("""QLabel{background-color:yellow;}""")
        if self.event_type[self.counter][1]=='control':
            self.frame_label.setStyleSheet("""QLabel{background-color:green;}""")
        if self.event_type[self.counter][1] == 'noise':
            self.frame_label.setStyleSheet("""QLabel{background-color:cyan;}""")

        self.frame_label.setText(f"Onset frame: {self.event_type[self.counter][0]}, Type is:  {self.event_type[self.counter][1]} ")


    def next_frame(self):
        self.counter +=1
        self.counter=min(len(self.event_type)-1,self.counter)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.event_type[self.counter][0])

        ret, frame = self.cap.read()
        self.fr_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.fr_image = cv2.resize(self.fr_image, (256, 256))
        self.image_item.setImage(self.fr_image)
        if self.fr_ROIs:
           self.frame_mask()
        if self.event_type[self.counter][1]=='reward':
            self.frame_label.setStyleSheet("""QLabel{background-color:blue;}""")
        if self.event_type[self.counter][1]=='airpuff':
            self.frame_label.setStyleSheet("""QLabel{background-color:red;}""")
        if self.event_type[self.counter][1]=='neutral':
            self.frame_label.setStyleSheet("""QLabel{background-color:yellow;}""")
        if self.event_type[self.counter][1]=='control':
            self.frame_label.setStyleSheet("""QLabel{background-color:green;}""")
        if self.event_type[self.counter][1]=='noise':
            self.frame_label.setStyleSheet("""QLabel{background-color:cyan;}""")
        self.frame_label.setText(f"Onset frame: {self.event_type[self.counter][0]}, Type is:  {self.event_type[self.counter][1]} ")


    def previous_frame(self):
        self.counter -= 1
        self.counter = max(0, self.counter)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.event_type[self.counter][0])

        ret, frame = self.cap.read()
        self.fr_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.fr_image = cv2.resize(self.fr_image, (256, 256))
        self.image_item.setImage(self.fr_image)
        if self.fr_ROIs:
           self.frame_mask()
        if self.event_type[self.counter][1]=='reward':
            self.frame_label.setStyleSheet("""QLabel{background-color:blue;}""")
        if self.event_type[self.counter][1]=='airpuff':
            self.frame_label.setStyleSheet("""QLabel{background-color:red;}""")
        if self.event_type[self.counter][1]=='neutral':
            self.frame_label.setStyleSheet("""QLabel{background-color:yellow;}""")
        if self.event_type[self.counter][1]=='control':
            self.frame_label.setStyleSheet("""QLabel{background-color:green;}""")
        if self.event_type[self.counter][1] == 'noise':
                self.frame_label.setStyleSheet("""QLabel{background-color:cyan;}""")
        self.frame_label.setText(f"Onset frame: {self.event_type[self.counter][0]}, Type is:  {self.event_type[self.counter][1]} ")

    def start_slide_show(self,next_prev):
        if next_prev=="next":
          self.next_timer.start(250)
        else:self.prev_timer.start(250)
    def stop_slide_show(self,next_prev):
        if next_prev == "next":
            self.next_timer.stop()
        else:
            self.prev_timer.stop()

    def addroi_function(self):
        self.fr_ROIs.append(FrameExtractROI.frame_extract_roi_class(parent=self))
        self.fr_nROIs +=1
        self.fr_ROIs[-1].position(self)




    def save_function(self):

        self.frame_dictionary_meta["ROI"]=[]
        if self.fr_ROIs:
            for roi in self.fr_ROIs:
                self.frame_dictionary_meta["ROI"].append([[roi.ROI.pos().y(), roi.ROI.pos().x()], [roi.ROI.size().y(), roi.ROI.size().x()], roi.roi_shape,roi.color])
            frame_meta = Path(self.video_path).stem
            np.save(os.path.join(split(self.video_path)[0], f"{frame_meta}_frame_Extraction_meta.npy"),self.frame_dictionary_meta)

            success_msg = QMessageBox()
            success_msg.setIcon(QMessageBox.Icon.Information)
            success_msg.setText("Successfully saved!")
            success_msg.setWindowTitle("Saved!")
            # success_msg.setStandardButtons(QMessageBox.StandaradButton.Ok)

            success_msg.exec_()
        else:
            failure_msg = QMessageBox()
            failure_msg.setIcon(QMessageBox.Icon.Critical)
            failure_msg.setText("There is no ROI to be Saved!")
            failure_msg.setWindowTitle("NO ROI!")
            # success_msg.setStandardButtons(QMessageBox.StandaradButton.Ok)

            failure_msg.exec_()



    def load_roi_Function(self):
        roi_path, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Open File Containing ROI)",
            directory="C:\LocalExpData",
            filter="ROI Files (*frame_Extraction_meta*.npy);"
        )
        if roi_path:
            try:
                roi_file = np.load(roi_path, allow_pickle=True).item()
                if self.fr_ROIs:
                    for roi in self.fr_ROIs:
                        self.video_view.removeItem(roi.ROI)
                self.fr_ROIs = []
                self.fr_nROIs = 0
                self.fr_pos_holder = []
                for roi in roi_file["ROI"]:
                    self.fr_ROIs.append(FrameExtractROI.frame_extract_roi_class(fr_roi_pos=roi[0], fr_roi_size=roi[1], fr_roi_shape=roi[2], fr_roi_color=roi[3],parent=self))
                    self.fr_nROIs += 1
                    self.fr_ROIs[-1].position(self)

            except:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Critical)
                msg.setText("No file Containing ROI")
                msg.setWindowTitle("Error!")
                # msg.setStandardButtons(QMessageBox.StandaradButton.Yes)
                msg.exec_()



    def load_frame_info(self):
        print("experiment_design",self.parent.experiment_design)
        print("experiment_length",self.parent.experiment_length)
        print("experiment_frames",self.parent.experiment_frames)
        print("frame exclude",self.parent.frame_exclude)
        self.exclusion_arr = []
        for f_ex0 in self.parent.frame_exclude:
            for f_ex in f_ex0[1]:
                self.exclusion_arr.extend(np.arange(f_ex[1][0] - 8, f_ex[1][1] + 8, 1))
        event_onset=[]
        for expl_ind,expl_val in enumerate(self.parent.experiment_length):
            event_onset.extend(self.parent.experiment_design[expl_ind][3][:expl_val])
        self.event_type=list(zip(self.parent.experiment_frames, event_onset))
        self.total_extract_frame=len(self.event_type)*sum(self.extraction_window)*self.capturing_fs
        self.individual_extract_frame=np.ones(len(self.event_type),int)*sum(self.extraction_window)*self.capturing_fs
        print("Total frames to be extracted: ",self.total_extract_frame)
        print("self.exclusion_arr",self.exclusion_arr)

    def frame_mask(self):
        self.masked_image0 = self.fr_image.copy()
        self.masked_out0 = np.zeros_like(self.fr_image)
        for i in range(len(self.fr_pos_holder)):
            if self.fr_pos_holder[i][2] == "Mask out":
                self.cut_image = self.fr_image[self.fr_pos_holder[i][1][0]:self.fr_pos_holder[i][1][-1],
                                 self.fr_pos_holder[i][0][0]:self.fr_pos_holder[i][0][-1]]

                mask_out_temp = [[self.fr_pos_holder[i][1][ii], self.fr_pos_holder[i][0][ii]] for ii in
                                 range(len(self.fr_pos_holder[i][0]))]
                for kk in mask_out_temp:
                    self.masked_out0[kk[0], kk[1]] = self.fr_image[kk[0], kk[1]]
                self.masked_image0 = self.masked_out0

        for i in range(len(self.fr_pos_holder)):
            if self.fr_pos_holder[i][2] != "Mask out":
                self.masked_image0[self.fr_pos_holder[i][1], self.fr_pos_holder[i][0]] = [0]
        self.roi_item.setImage(self.masked_image0)


    def extract_frames_function(self,mess=None):
        if mess !='progress':
            self.extraction_counter = 0
        if  self.extraction_counter == 0:
            self.process_duration=time()
            print("Started")
            self.progressupdate = Progress_bar_extraction.Progress_bar_extraction(self)
        if self.fr_extraction_mode.currentText()=='All' and self.extraction_counter == 0:
            self.load_video()

        frame_extraction_path=Path(self.video_path).parent
        camera_detection=Path(self.video_path).stem
        camera_detection=camera_detection.split('_')[0]
        frame_directory=os.path.join(frame_extraction_path, 'Frame Extraction')
        if not os.path.exists(frame_directory):
           os.makedirs(frame_directory, exist_ok=True)
        event_directory=os.path.join(frame_directory,f"{camera_detection}_{self.event_type[self.counter][1]}")
        os.makedirs(event_directory, exist_ok=True)
        self.extract_frames.setEnabled(False)
        self.fr_extraction_mode.setEnabled(False)
        self.save_button.setEnabled(False)
        self.addroi_button.setEnabled(False)
        self.loadroi_button.setEnabled(False)
        self.fr_combo_box.setEnabled(False)
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.preview_btn.setEnabled(False)
        progress_thread=Progress_bar_extraction.progress_thread(event_directory,parent=self)
        progress_thread.log_signal.connect(self.update_progress)
        progress_thread.start()


    def update_progress(self,message):
        self.progressupdate.update_progress(message)


    # def check_box_position(self):
    #     scene=self.video_view.mapToView(QPointF(self.video_view.width(),self.video_view.height()))
    #     chck_view=self.check_view.boundingRect()
    #     x=scene.x()-chck_view.width()
    #     y = scene.y() - chck_view.height()
    #     self.check_view.setPos(x,y)

    def preview_function(self):
        if self.fr_ROIs:
           preview_instance=Frame_Extraction_Preview.frame_extraction_preview(parent=self)
        else:
            no_preview = QMessageBox()
            no_preview.setIcon(QMessageBox.Icon.Critical)
            no_preview.setText("No ROI, No Preview!")
            no_preview.setWindowTitle("No ROI, No Preview!")
            no_preview.exec_()
