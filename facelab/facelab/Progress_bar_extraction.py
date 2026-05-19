import os
from pathlib import Path
from time import time

import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from PyQt5.QtWidgets import QVBoxLayout, QProgressBar, QMessageBox

import facelab.prediction_check_gui as prediction_check_gui


class progress_thread(QThread):
    log_signal = pyqtSignal(list)

    def __init__(self,event_directory,parent=None):
        super().__init__(parent)
        # self.log_signal = pyqtSignal(int)
        self.parent=parent
        self.event_directory=event_directory



    def run(self):
        if self.parent.cap_extract is not None:
            self.parent.cap_extract.release()
        print(int(self.parent.fr_extraction_window.currentText()[1]),int(self.parent.fr_extraction_window.currentText()[3]))
        self.extraction_frames=np.arange(self.parent.event_type[self.parent.counter][0]-(self.parent.capturing_fs*int(self.parent.fr_extraction_window.currentText()[1])),self.parent.event_type[self.parent.counter][0]+(self.parent.capturing_fs*int(self.parent.fr_extraction_window.currentText()[3])),1)
        self.parent.cap_extract=cv2.VideoCapture(self.parent.video_path)
        self.parent.fr_progress_overall = len(self.extraction_frames)
        self.image_holder=[]
        self.name_holder=[]
        for f_r_ind,f_r_val in enumerate(self.extraction_frames):

            if f_r_val not in self.parent.exclusion_arr:
                self.parent.cap_extract.set(cv2.CAP_PROP_POS_FRAMES,f_r_val)
                extract_ret, extract_frame = self.parent.cap_extract.read()
                extract_image= cv2.cvtColor(extract_frame, cv2.COLOR_BGR2GRAY)
                extract_image = cv2.resize(extract_image, (256, 256))
                self.image_holder.append(extract_image)
                self.name_holder.append([f_r_ind,f_r_val])
        self.parent.fr_progress_overall = len(self.image_holder)
        maseked_extract_chunk_image0=np.array(self.image_holder.copy())
        masked_extract_chunk_out0=np.zeros_like(self.image_holder)
        for i in range(len(self.parent.fr_pos_holder)):
            if self.parent.fr_pos_holder[i][2] == "Mask out":
                # mask_out_temp_chunk_extract = [[self.parent.fr_pos_holder[i][1][ii], self.parent.fr_pos_holder[i][0][ii]] for ii in range(len(self.parent.fr_pos_holder[i][0]))]

                masked_extract_chunk_out0[:,self.parent.fr_pos_holder[i][1], self.parent.fr_pos_holder[i][0]] = np.array(self.image_holder)[:,self.parent.fr_pos_holder[i][1], self.parent.fr_pos_holder[i][0]]
                maseked_extract_chunk_image0 = masked_extract_chunk_out0


        for i in range(len(self.parent.fr_pos_holder)):
            if self.parent.fr_pos_holder[i][2] != "Mask out":
                maseked_extract_chunk_image0[:,self.parent.fr_pos_holder[i][1], self.parent.fr_pos_holder[i][0]] = [0]

        for s_image_ind,s_image_val in enumerate(maseked_extract_chunk_image0):
            if self.name_holder[s_image_ind][1]>=self.parent.event_type[self.parent.counter][0]:
               cv2.imwrite(os.path.join(self.event_directory, f"{self.parent.event_type[self.parent.counter][1]} ({self.parent.event_type[self.parent.counter][0]})_{self.name_holder[s_image_ind][1]}_after.jpg"),s_image_val)
            else:cv2.imwrite(os.path.join(self.event_directory, f"{self.parent.event_type[self.parent.counter][1]} ({self.parent.event_type[self.parent.counter][0]})_{self.name_holder[s_image_ind][1]}_before.jpg"),s_image_val)
            if self.parent.fr_extraction_mode.currentText() != 'All':
                self.log_signal.emit([s_image_ind, int(100 * s_image_ind / (self.parent.fr_progress_overall - 1))])


        if self.parent.fr_extraction_mode.currentText() == 'All':
            self.log_signal.emit([f_r_ind, int(100 * (f_r_ind + sum(self.parent.individual_extract_frame[:self.parent.extraction_counter])) / (self.parent.total_extract_frame-1))])





class Progress_bar_extraction(QtWidgets.QDialog):
    def __init__(self,parent=None,gui=None,main_gui=None):
        super().__init__(parent)
        self.setWindowTitle("Progress...")
        #
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.Dialog)
        # self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        # self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.progress_layout=QVBoxLayout()
        self.setFixedSize(800, 200)
        self.setLayout(self.progress_layout)
        self.progress_button=QProgressBar()
        self.progress_button.setFixedHeight(150)
        self.progress_layout.addWidget(self.progress_button)
        self.progress_button.setRange(0,100)
        self.progress_button.setValue(0)
        self.parent=parent
        self.gui=gui
        self.main_gui=main_gui
        self.show()

    def update_progress(self,message,training=False,destination_path=None):

        if training==False:
            self.progress_button.setValue(message[1])
            extraction_mode=self.parent.fr_extraction_mode.currentText()
            if extraction_mode=="All" and message[0]==self.parent.individual_extract_frame[self.parent.extraction_counter]-1 and message[1] !=100:
                self.parent.extraction_counter +=1
                self.parent.next_frame()
                self.parent.extract_frames_function(mess='progress')
            elif message[1] ==100:
                self.close()
                self.parent.extract_frames.setEnabled(True)
                self.parent.fr_extraction_mode.setEnabled(True)
                self.parent.save_button.setEnabled(True)
                self.parent.addroi_button.setEnabled(True)
                self.parent.loadroi_button.setEnabled(True)
                self.parent.fr_combo_box.setEnabled(True)
                self.parent.previous_button.setEnabled(True)
                self.parent.next_button.setEnabled(True)
                self.parent.preview_btn.setEnabled(True)
                print("Duration: ", time()-self.parent.process_duration)
        else:
            self.progress_button.setValue(message)
            if message==100:
                prediction_check_gui.prediction_check(self.gui,self.main_gui,destination_path=destination_path)
                self.gui.close()
                self.close()

                # prediction_check_gui.prediction_check(self.parent)
                #
                # msg=QMessageBox()
                # msg.setIcon(QMessageBox.Icon.Information)
                # msg.setText(f"'{Path(destination_path).stem}' Trained and Saved!")
                # msg.setWindowTitle(Path(destination_path).stem)
                # msg.setStandardButtons(QMessageBox.StandardButton.Ok)
                # msg.exec_()


        # if message==100:
        #     self.close()
        #     self.parent.extract_frames.setEnabled(True)






