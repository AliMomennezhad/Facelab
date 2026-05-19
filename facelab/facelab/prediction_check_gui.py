import sys

import cv2
import numpy as np
import torch
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QApplication, QPushButton
import  pyqtgraph as pg
from cv2 import imread
from pyqtgraph.Qt import QtGui

import facelab.finetune_preparation as finetune_preparation
import facelab.keypoint_prediction as keypoint_prediction
from facelab.keypoint_model_training import Keypoint_Class


class prediction_check(QDialog):
    def __init__(self,parent=None,main_gui=None,destination_path=None):
        super().__init__(parent)
        self.setStyleSheet("QDialog {background:'white';}")
        self.setWindowFlags(
            Qt.Window |  # Basic window with title bar
            Qt.WindowMinimizeButtonHint |  # Enable minimize button
            Qt.WindowCloseButtonHint  # Enable close button
        )
        self.setWindowTitle("Prediction Check")
        self.setGeometry(100, 0, 1600, 900)
        QCoreApplication.setApplicationName("Prediction Check GUI")

        centerPoint = QGuiApplication.primaryScreen().availableGeometry().center()
        qtRectangle = self.frameGeometry()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())
        pg.setConfigOptions(imageAxisOrder='row-major')
        # Initialize the layout
        center_layout = QVBoxLayout()
        self.setLayout(center_layout)


        self.image_window=[pg.GraphicsLayoutWidget() for i in range(12)]
        self.image_view=[im_win.addViewBox(lockAspect=True,invertY=True) for im_win in self.image_window]

        self.image_item=[pg.ImageItem() for i in range(12)]
        self.scatter_item = [pg.ScatterPlotItem() for i in range(12)]
        for im_view_ind,im_view_val in enumerate(self.image_view):
            im_view_val.addItem(self.image_item[im_view_ind])
            im_view_val.addItem(self.scatter_item[im_view_ind])

        layouts=[QHBoxLayout() for i in range(12//3)]
        for layout_ind,layout_val in enumerate(layouts):
            layout_val.addWidget(self.image_window[layout_ind*3+0])
            layout_val.addWidget(self.image_window[layout_ind*3+1])
        for layout in layouts:
            center_layout.addLayout(layout)
        #
        self.save_btn=QPushButton("Save Model")
        self.continue_btn = QPushButton("Continue Training")
        self.btn_layout=QHBoxLayout()
        self.btn_layout.addWidget(self.save_btn)
        self.btn_layout.addWidget(self.continue_btn)
        center_layout.addLayout(self.btn_layout)

        self.save_btn.clicked.connect(self.save_function)
        self.continue_btn.clicked.connect(self.continue_function)
        self.parent = parent
        self.main_gui=main_gui
        self.destination_path=destination_path
        self.check_keypoint_creation()
        self.show()

    def save_function(self):
        torch.save({"model_state_dict": self.parent.model.state_dict(), "optimizer_state_dict": self.parent.optimizer.state_dict()},self.destination_path)
        self.main_gui.keypoint_combo_item()
        self.close()

    def continue_function(self):
        finetune_preparation.finetune_prep(parent=self.main_gui,check=True,check_parent=self.parent,model=self.parent.model)
        self.close()

    def check_keypoint_creation(self):
        cap = cv2.VideoCapture(self.main_gui.video_path[0])
        self.rand_frames = np.random.choice(self.main_gui.overall_frame, 12, replace=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.frame_cunck=[]
        for r_f in self.rand_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, r_f)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (256, 256))
            self.frame_cunck.append(frame)
        x_key, y_key, likelihood_key = keypoint_prediction.predict_net(self.frame_cunck,self.parent.model, self.device)
        for f_ind,f_val in enumerate(self.frame_cunck):
            self.image_item[f_ind].setImage(f_val)
            self.scatter_item[f_ind].setData(pos=[[y_key[f_ind][0],x_key[f_ind][0]],[y_key[f_ind][1],x_key[f_ind][1]],[y_key[f_ind][2],x_key[f_ind][2]]],hoverable=True,hoverSymbol="x",pxMode=True,hoverBrush="r",symbolPen=pg.mkPen(color=(255, 255, 255)), size=6, symbol='o',brush=[QtGui.QColor('DarkOrange'),QtGui.QColor('blue'),QtGui.QColor('yellow')])

