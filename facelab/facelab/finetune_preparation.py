import glob
import os
from pathlib import Path

from PyQt5.QtCore import QCoreApplication, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QGuiApplication, QFont
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QLineEdit, QSpinBox, QPushButton
import pyqtgraph as pg

import facelab.finetune_prediction as finetune_prediction
import facelab.keypoint_fine_tuning as keypoint_fine_tuning

import facelab.train_model as train_model
from facelab.video_info import get_frame_number


class finetune_prep(QDialog):
    def __init__(self,parent=None,check=None,check_parent=None,model=None):
        super().__init__(parent)
        self.setStyleSheet("QDialog {background:'lightblue';}")
        self.setWindowFlags(
            Qt.Window |  # Basic window with title bar
            Qt.WindowMinimizeButtonHint |  # Enable minimize button
            Qt.WindowCloseButtonHint  # Enable close button
        )
        self.setWindowTitle("Fine Tune Parameters")
        self.setGeometry(300, 100, 800, 600)
        QCoreApplication.setApplicationName("Keypoint Fine-Tuning GUI")

        centerPoint = QGuiApplication.primaryScreen().availableGeometry().center()
        qtRectangle = self.frameGeometry()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())
        pg.setConfigOptions(imageAxisOrder='row-major')

        center_layout = QVBoxLayout()
        self.setLayout(center_layout)
        self.parent = parent
        self.check=check
        self.model=model
        self.check_parent=check_parent

        self.frame_num_layout=QHBoxLayout()
        self.frame_label=QLabel("Num of Frames: ")
        self.frame_label.setStyleSheet("""QLabel{background:'white';}""")
        self.frame_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.frame_label.setFixedWidth(250)
        self.frame_val=QSpinBox()
        self.frame_val.setFont(QFont("Arial", 12, QFont.Bold))
        self.frame_label.setStyleSheet("""QSpinBox{background:'white';}""")
        self.frame_val.setFixedWidth(250)
        self.frame_val.setRange(1,100)
        self.frame_val.setValue(25)
        self.frame_num_layout.addWidget(self.frame_label)
        self.frame_num_layout.addWidget(self.frame_val)
        self.intitial_model_layout=QHBoxLayout()
        self.initial_model_label=QLabel("Select Initial Model: ")
        self.initial_model_label.setStyleSheet("""QLabel{background:'white';}""")
        self.initial_model_label.setFont(QFont("Arial",12,QFont.Bold))
        self.initial_model_label.setFixedWidth(250)
        self.initial_model_combo=QComboBox()
        if self.check is None:
           self.model_directory(self.initial_model_combo)
        else:
            self.frame_label.setText("add frames: ")
            self.frame_val.setValue(5)
        self.initial_model_combo.setFont(QFont("Arial",12,QFont.Bold))
        self.initial_model_combo.setFixedWidth(250)
        self.intitial_model_layout.addWidget(self.initial_model_label)
        self.intitial_model_layout.addWidget(self.initial_model_combo)

        self.target_model_layout=QHBoxLayout()
        self.target_model_label = QLabel("Target Model: ")
        self.target_model_label.setFixedWidth(250)
        self.target_model_label.setStyleSheet("""QLabel{background:'white';}""")
        self.target_model_label.setFont(QFont("Arial",12,QFont.Bold))
        self.target_model_box = QLineEdit()
        self.target_model_box.setFixedWidth(250)
        self.target_model_box.setFont(QFont("Arial",12,QFont.Bold))
        self.target_model_layout.addWidget(self.target_model_label)
        self.target_model_layout.addWidget(self.target_model_box)

        self.miscelaneous_param_layout=QHBoxLayout()
        self.lr_label=QLabel("Learning Rate: ")
        self.lr_label.setFont(QFont("Arial",12,QFont.Bold))
        self.lr_label.setFixedWidth(150)
        self.lr_values=QComboBox()
        self.lr_values.setFixedWidth(150)
        self.lr_values.addItems(["0.001","0.0001","0.00001","0.000001","0.0000001"])
        self.lr_values.setCurrentIndex(1)
        self.lr_values.setFont(QFont("Arial",12,QFont.Bold))

        self.epoch_label = QLabel("Epochs: ")
        self.epoch_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.epoch_label.setFixedWidth(100)
        self.epoch_val = QSpinBox()
        self.epoch_val.setFont(QFont("Arial", 12, QFont.Bold))
        self.epoch_val.setStyleSheet("""QSpinBox{background:'white';}""")
        self.epoch_val.setFixedWidth(100)
        self.epoch_val.setRange(1, 1000)
        self.epoch_val.setValue(40)

        self.miscelaneous_param_layout.addWidget(self.lr_label)
        self.miscelaneous_param_layout.addWidget(self.lr_values)
        self.miscelaneous_param_layout.addWidget(self.epoch_label)
        self.miscelaneous_param_layout.addWidget(self.epoch_val)

        self.proceed_btn_layout=QHBoxLayout()
        self.proceed_btn=QPushButton("Proceed")
        self.proceed_btn.setFixedWidth(250)
        self.proceed_btn.setFont(QFont("Arial",12,QFont.Bold))
        self.proceed_btn_layout.addWidget(self.proceed_btn)
        center_layout.addLayout(self.frame_num_layout)
        if self.check is None:
            center_layout.addLayout(self.intitial_model_layout)
            center_layout.addLayout(self.target_model_layout)
        center_layout.addLayout(self.miscelaneous_param_layout)
        center_layout.addLayout(self.proceed_btn_layout)

        self.proceed_btn.clicked.connect(self.proceed_function)
        self.show()

    def proceed_function(self):
        self.parent.frame_val=self.frame_val.value()
        self.parent.epoch_val = self.epoch_val.value()
        self.parent.lr_values = float(self.lr_values.currentText())
        if self.check is None:
            self.parent.x_key_arr=[]
            self.parent.y_key_arr = []
            self.parent.like_key_arr = []
            self.parent.frame_set_arr = []
            self.parent.frame_data_arr = []
            self.parent.target_model_box=self.target_model_box
            self.parent.initial_model_combo = self.initial_model_combo
            if self.target_model_box.text()=='':
                text="model_name.pt"
            else:text=self.target_model_box.text()+'.pt'
            self.parent.destination_path=os.path.join(self.model_dir,text)
        else:
            self.parent.destination_path=self.check_parent.parent.destination_path

        self.close()
        # fine_pred_instance=finetune_prediction.finetune_prediction(parent=self.parent,model=self.model)
        self.finetune_thread=finetune_prediction.finetune_thread(parent=self.parent,model=self.model)
        self.finetune_thread.log_signal_finetune.connect(self.call_finetune_gui)
        self.finetune_thread.start()

        # x_key,y_key,likelihood_key,frame_set,frame_data=fine_pred_instance.finetune_data()
        # self.parent.x_key_arr=[*x_key,*self.parent.x_key_arr]
        # self.parent.y_key_arr = [*y_key, *self.parent.y_key_arr]
        # self.parent.like_key_arr = [*likelihood_key, *self.parent.like_key_arr]
        # self.parent.frame_set_arr = [*frame_set, *self.parent.frame_set_arr]
        # self.parent.frame_data_arr = [*frame_data, *self.parent.frame_data_arr]
        # keypoint_fine_tuning.DraggablePointDialog(self.parent.x_key_arr,self.parent.y_key_arr,self.parent.like_key_arr,self.parent.frame_set_arr,self.parent.frame_data_arr,parent=self.parent)

    def call_finetune_gui(self,message):
        if message=="finished":
           keypoint_fine_tuning.DraggablePointDialog(self.parent.x_key_arr, self.parent.y_key_arr,
                                                  self.parent.like_key_arr, self.parent.frame_set_arr,
                                                  self.parent.frame_data_arr, parent=self.parent)
        else:
            get_frame_number(self.parent, self.parent.video_path[0], int(float(message)*self.parent.overall_frame))


    def model_directory(self,widget):
        self.model_dir=r"C:\LocalExpData\models"
        os.makedirs(self.model_dir,exist_ok=True)
        model_list0=glob.glob(os.path.join(self.model_dir,'*.pt'))
        self.parent.model_name=["basic_model.pt"]
        self.parent.model_name.extend([m_l for m_l in model_list0 if m_l != self.parent.model_name[0]])
        combo_name=[Path(m_l).stem for m_l in self.parent.model_name]
        widget.addItems(combo_name)


class fine_tune_thread(QThread):
    log_signal = pyqtSignal(int)
    def __init__(self,keypoint_saving_path,lr_values ,destination_path=None,model_path=None,parent=None,epoch=None):
        super().__init__()
        self.parent=parent
        self.keypoint_saving_path=keypoint_saving_path
        self.lr_values=lr_values
        self.destination_path=destination_path
        self.model_path=model_path
        self.epoch=epoch



    def run(self):
        train_model.initiate_training(self.keypoint_saving_path,self.lr_values ,destination_path=self.destination_path,model_path=self.model_path,parent=self.parent,epoch=self.epoch,log_signal=self.log_signal)



