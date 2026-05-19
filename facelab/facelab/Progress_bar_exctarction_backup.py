import os
from time import time

import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from PyQt5.QtWidgets import QVBoxLayout, QProgressBar


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
        extraction_frames=np.arange(self.parent.event_type[self.parent.counter][0]-(self.parent.capturing_fs*self.parent.extraction_window[0]),self.parent.event_type[self.parent.counter][0]+(self.parent.capturing_fs*self.parent.extraction_window[1]),1)
        self.parent.cap_extract=cv2.VideoCapture(self.parent.video_path)
        self.parent.fr_progress_overall = len(extraction_frames)
        for f_r_ind,f_r_val in enumerate(extraction_frames):

            if f_r_val not in self.parent.exclusion_arr:
                self.parent.cap_extract.set(cv2.CAP_PROP_POS_FRAMES,f_r_val)
                extract_ret, extract_frame = self.parent.cap_extract.read()
                extract_image= cv2.cvtColor(extract_frame, cv2.COLOR_BGR2GRAY)
                extract_image = cv2.resize(extract_image, (256, 256))

                masked_extract_image0 = extract_image.copy()
                masked_extract_out0 = np.zeros_like(extract_image)
                for i in range(len(self.parent.fr_pos_holder)):
                    if self.parent.fr_pos_holder[i][2] == "Mask out":
                        mask_out_temp_extract = [[self.parent.fr_pos_holder[i][1][ii], self.parent.fr_pos_holder[i][0][ii]] for ii in
                                         range(len(self.parent.fr_pos_holder[i][0]))]
                        for kk_extract in mask_out_temp_extract:
                            masked_extract_out0[kk_extract[0], kk_extract[1]] = extract_image[kk_extract[0], kk_extract[1]]
                        masked_extract_image0= masked_extract_out0

                for i in range(len(self.parent.fr_pos_holder)):
                    if self.parent.fr_pos_holder[i][2] != "Mask out":
                        masked_extract_image0[self.parent.fr_pos_holder[i][1], self.parent.fr_pos_holder[i][0]] = [0]
                if f_r_val>=self.parent.event_type[self.parent.counter][0]:
                   cv2.imwrite(os.path.join(self.event_directory, f"{self.parent.event_type[self.parent.counter][1]} ({self.parent.event_type[self.parent.counter][0]})_{f_r_val}_after.jpg"),masked_extract_image0)
                else:cv2.imwrite(os.path.join(self.event_directory, f"{self.parent.event_type[self.parent.counter][1]} ({self.parent.event_type[self.parent.counter][0]})_{f_r_val}_before.jpg"),masked_extract_image0)

            if self.parent.fr_extraction_mode.currentText() == 'All':
                self.log_signal.emit([f_r_ind, int(100 * (f_r_ind + sum(self.parent.individual_extract_frame[:self.parent.extraction_counter])) / (
                                                       self.parent.total_extract_frame-1))])
            else:
                self.log_signal.emit([f_r_ind,int(100 * f_r_ind / (self.parent.fr_progress_overall - 1))])





class Progress_bar_extraction(QtWidgets.QDialog):
    def __init__(self,parent=None):
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
        self.show()

    def update_progress(self,message):
        self.progress_button.setValue(message[1])
        extraction_mode=self.parent.fr_extraction_mode.currentText()
        if extraction_mode=="All" and message[0]==self.parent.individual_extract_frame[self.parent.extraction_counter]-1 and message[1] !=100:
            self.parent.extraction_counter +=1
            self.parent.next_frame()
            self.parent.extract_frames_function(mess='progress')
        elif message[1] ==100:
            self.close()
            self.parent.extract_frames.setEnabled(True)
            print("Duration: ", time() - self.parent.process_duration)


        # if message==100:
        #     self.close()
        #     self.parent.extract_frames.setEnabled(True)






