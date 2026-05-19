import cv2
import numpy as np
from PyQt5.QtCore import Qt, QSize, QCoreApplication, QPointF, QTimer
from PyQt5.QtGui import QIcon, QGuiApplication, QFont
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QCheckBox, QGraphicsProxyWidget, QGraphicsItem, QPushButton, \
    QStyle, QHBoxLayout, QLabel

from pyqtgraph.Qt import QtWidgets
import pyqtgraph as pg


class frame_extraction_preview(QtWidgets.QDialog):
    def __init__(self,parent=None):
        super().__init__(parent)

        self.setWindowFlags(
            Qt.Window |  # Basic window with title bar
            Qt.WindowMinimizeButtonHint |  # Enable minimize button
            Qt.WindowCloseButtonHint  # Enable close button
        )

        self.setGeometry(300, 200, 1250, 800)
        app_icon = QIcon()
        icon_path = r'keypoint_icon.png'
        app_icon.addFile(icon_path, QSize(16, 16))
        app_icon.addFile(icon_path, QSize(24, 24))
        app_icon.addFile(icon_path, QSize(32, 32))
        app_icon.addFile(icon_path, QSize(48, 48))
        app_icon.addFile(icon_path, QSize(96, 96))
        app_icon.addFile(icon_path, QSize(256, 256))
        QApplication.setWindowIcon(app_icon)

        centerPoint = QGuiApplication.primaryScreen().availableGeometry().center()
        qtRectangle = self.frameGeometry()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())
        pg.setConfigOptions(imageAxisOrder='row-major')

        main_layout=QVBoxLayout()
        self.setLayout(main_layout)

        self.image_window=pg.GraphicsLayoutWidget()
        self.image_view=self.image_window.addViewBox(lockAspect=True,invertY=True)
        self.image_item=pg.ImageItem()
        self.image_view.addItem(self.image_item)
        self.check_item = QCheckBox('Excluded')
        self.check_item.setStyleSheet("""QCheckBox{background-color:black;color:white}""")
        self.check_item.setFont(QFont("Arial", 14, QFont.Bold))
        self.check_view = QGraphicsProxyWidget()
        self.check_view.setWidget(self.check_item)
        self.check_view.setParentItem(self.image_view)
        self.check_view.setZValue(1000)
        self.image_view.addItem(self.check_view, ignoreBounds=True)
        self.image_view.sigResized.connect(self.check_box_position)
        self.check_view.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.check_box_position()


        self.btn_layout=QHBoxLayout()

        self.frame_label=QLabel("Frame: ")
        self.frame_label.setFont(QFont("Arial",14,QFont.Bold))
        self.frame_label.setFixedHeight(50)


        self.frame_display = QLabel(self)
        self.frame_display.setFont(QFont("Arial", 14, QFont.Bold))
        self.frame_display.setFixedHeight(50)

        self.onset_frame_display = QLabel(self)
        self.onset_frame_display.setFont(QFont("Arial", 10, QFont.Bold))
        self.onset_frame_display.setFixedHeight(50)


        self.next_btn=QPushButton(self)
        self.next_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_ArrowRight))
        self.next_btn.setFixedHeight(50)

        self.prev_btn = QPushButton(self)
        self.prev_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_ArrowLeft))
        self.prev_btn.setFixedHeight(50)

        self.pause_btn = QPushButton(self)
        self.pause_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaPause))
        self.pause_btn.setFixedHeight(50)

        self.play_btn = QPushButton(self)
        self.play_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_btn.setFixedHeight(50)

        self.btn_layout.addWidget(self.play_btn)
        self.btn_layout.addWidget(self.pause_btn)
        self.btn_layout.addWidget(self.prev_btn)
        self.btn_layout.addWidget(self.next_btn)
        self.btn_layout.addWidget(self.frame_label)
        self.btn_layout.addWidget(self.frame_display)
        self.btn_layout.addWidget(self.onset_frame_display)


        main_layout.addWidget(self.image_window)
        main_layout.addLayout(self.btn_layout)




        self.next_timer=QTimer()
        self.next_timer.timeout.connect(self.next_frame)
        self.prev_timer = QTimer()
        self.prev_timer.timeout.connect(self.prev_frame)
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)

        self.play_btn.clicked.connect(lambda:self.start_timer("play"))
        self.pause_btn.clicked.connect(lambda: self.stop_timer("play"))




        self.next_btn.clicked.connect(self.next_frame)
        self.next_btn.pressed.connect(lambda: self.start_timer("next"))
        self.next_btn.released.connect(lambda: self.stop_timer("next"))


        self.prev_btn.clicked.connect(self.prev_frame)
        self.prev_btn.pressed.connect(lambda: self.start_timer("prev"))
        self.prev_btn.released.connect(lambda: self.stop_timer("prev"))
        self.check_item.clicked.connect(self.check_toggle)

        self.parent = parent
        self.preview_cap=None
        self.preview_conter=0

        self.load_frames()
        self.show()

    def check_box_position(self):
        scene = self.image_view.mapToView(QPointF(self.image_view.width(), self.image_view.height()))
        chck_view = self.check_view.boundingRect()
        x = scene.x() - chck_view.width()
        y = scene.y() - chck_view.height()
        self.check_view.setPos(x, y)

    def load_frames(self):
        if self.preview_cap is not None:
            self.preview_cap.release()
        self.extraction_frames=np.arange(self.parent.event_type[self.parent.counter][0]-(self.parent.capturing_fs*self.parent.extraction_window[0]),self.parent.event_type[self.parent.counter][0]+(self.parent.capturing_fs*self.parent.extraction_window[1]),1)
        self.preview_cap = cv2.VideoCapture(self.parent.video_path)
        self.total_review_frames = len(self.extraction_frames)
        self.selected_frame=self.extraction_frames[0]
        self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES,self.selected_frame)
        ret,frame=self.preview_cap.read()
        self.preview_image=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        self.preview_image=cv2.resize(self.preview_image,(256,256))
        if self.parent.event_type[self.parent.counter][1]=='reward':
            self.onset_frame_display.setStyleSheet("""QLabel{background-color:blue;}""")
        if self.parent.event_type[self.parent.counter][1]=='airpuff':
            self.onset_frame_display.setStyleSheet("""QLabel{background-color:red;}""")
        if self.parent.event_type[self.parent.counter][1]=='neutral':
            self.onset_frame_display.setStyleSheet("""QLabel{background-color:yellow;}""")
        if self.parent.event_type[self.parent.counter][1]=='control':
            self.onset_frame_display.setStyleSheet("""QLabel{background-color:green;}""")
        self.onset_frame_display.setText(f"{self.parent.event_type[self.parent.counter][1]}-onset frame: {self.parent.event_type[self.parent.counter][0]}")
        QCoreApplication.setApplicationName(f"Preview for {self.parent.event_type[self.parent.counter][1]}(onset frame: {self.parent.event_type[self.parent.counter][0]})")
        self.mask_image()

    def mask_image(self):
        self.masked_image0 = self.preview_image.copy()
        self.masked_out0 = np.zeros_like(self.preview_image)
        for i in range(len(self.parent.fr_pos_holder)):
            if self.parent.fr_pos_holder[i][2] == "Mask out":
                self.cut_image = self.preview_image[self.parent.fr_pos_holder[i][1][0]:self.parent.fr_pos_holder[i][1][-1],
                                 self.parent.fr_pos_holder[i][0][0]:self.parent.fr_pos_holder[i][0][-1]]

                mask_out_temp = [[self.parent.fr_pos_holder[i][1][ii], self.parent.fr_pos_holder[i][0][ii]] for ii in range(len(self.parent.fr_pos_holder[i][0]))]
                for kk in mask_out_temp:
                    self.masked_out0[kk[0], kk[1]] = self.preview_image[kk[0], kk[1]]
                self.masked_image0 = self.masked_out0

        for i in range(len(self.parent.fr_pos_holder)):
            if self.parent.fr_pos_holder[i][2] != "Mask out":
                self.masked_image0[self.parent.fr_pos_holder[i][1], self.parent.fr_pos_holder[i][0]] = [0]
        self.image_item.setImage(self.masked_image0)
        self.frame_display.setText(str(self.selected_frame))
        if self.selected_frame in self.parent.exclusion_arr:
            self.check_item.setChecked(True)
        else:self.check_item.setChecked(False)



    def start_timer(self,text):
        if text=="next":
            self.next_timer.start(100)
        elif text=="prev":
            self.prev_timer.start(40)
        else:
            self.play_timer.start(30)
            self.play_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.prev_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)



    def stop_timer(self,text):
        if text=="next":
            self.next_timer.stop()
        elif text=="prev":
            self.prev_timer.stop()
        else:
            self.play_timer.stop()
            self.play_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            self.prev_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)


    def next_frame(self):
        self.preview_conter +=1
        self.preview_conter=min(self.preview_conter,self.total_review_frames-1)
        if self.preview_conter==self.total_review_frames-1:
            self.stop_timer("play")
        self.selected_frame=self.extraction_frames[self.preview_conter]
        self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, self.selected_frame)
        ret, frame = self.preview_cap.read()
        self.preview_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self.preview_image = cv2.resize(self.preview_image, (256, 256))
        self.mask_image()


    def prev_frame(self):
        self.preview_conter -= 1
        self.preview_conter = max(self.preview_conter, 0)
        self.selected_frame = self.extraction_frames[self.preview_conter]
        self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, self.selected_frame)
        ret, frame = self.preview_cap.read()
        self.preview_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self.preview_image = cv2.resize(self.preview_image, (256, 256))
        self.mask_image()


    def check_toggle(self):

        if self.check_item.isChecked():
            if self.selected_frame not in self.parent.exclusion_arr:
               self.parent.exclusion_arr.extend([self.selected_frame])
        else:
            if self.selected_frame in self.parent.exclusion_arr:
                self.parent.exclusion_arr.remove(self.selected_frame)

