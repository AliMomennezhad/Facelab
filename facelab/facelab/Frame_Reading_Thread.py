import os
from time import time

import cv2
import h5py
import numpy as np
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from tqdm import tqdm
from os.path import split
from pathlib import Path

import facelab.keypoint_prediction as keypoint_prediction
import facelab.video_info as video_info
from facelab.MainMenu import keypoint_check
from facelab.keypoint_model_training import Keypoint_Class


class FrameReaderThread(QThread):
    log_signal = pyqtSignal(str)

    def __init__(self, video_path,overal_frame,blink_arr=None,key_model=None, parent=None):
        super().__init__()
        self.video_path = video_path
        self.chunk_size=500
        self.blink_arr=blink_arr
        self.chunk=[]
        self.overal_frame=overal_frame
        self.parent=parent
        self.key_model=key_model


    def run(self):
        key_point_meta_path=Path(self.video_path).parent
        if self.parent.keypoint_check.isChecked():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Keypoint_Class().to(device)
            if self.key_model is not None:
                print("This model was selected: ",self.key_model)
                model.load_state_dict(torch.load(self.key_model, map_location=device)["model_state_dict"])
            else:
                 print("Default model was selected")
                 model.load_state_dict(torch.load(r"C:\LocalExpData\blinking2\2025-04-22_1\concatenated\mymodel.pt",map_location=device))
            self.keypoint_data_x=[]
            self.keypoint_data_y=[]
            self.keypoint_likelihood = []
            start_time=time()
            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret=True
            while ret:
                for _ in tqdm(range(self.overal_frame + 1), desc="frames"):
                    ret, frame0 = cap.read()
                    if ret:
                        self.keypoint_frame=cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
                        self.keypoint_frame = cv2.resize(self.keypoint_frame, (256, 256))
                        if len(self.chunk) < self.chunk_size:
                            self.chunk.append(self.keypoint_frame)
                        if len(self.chunk) == self.chunk_size:
                            x_key,y_key,likelihood_key=keypoint_prediction.predict_net(self.chunk,model,device)
                            self.keypoint_data_x.extend(x_key)
                            self.keypoint_data_y.extend(y_key)
                            self.keypoint_likelihood.extend(likelihood_key)
                            self.log_signal.emit(str(len(self.keypoint_data_x)))
                            self.parent.frame_num.setText(str(len(self.keypoint_data_x)))
                            self.parent.slider_frame.setValue(len(self.keypoint_data_x))
                            video_info.get_frame_number(self.parent, self.video_path, len(self.keypoint_data_x))
                            print(len(self.keypoint_data_x))
                            self.chunk = []
                    elif not ret and len(self.chunk) != 0:
                        x_key,y_key,likelihood_key=keypoint_prediction.predict_net(self.chunk,model,device)
                        self.keypoint_data_x.extend(x_key)
                        self.keypoint_data_y.extend(y_key)
                        self.keypoint_likelihood.extend(likelihood_key)
                        self.chunk = []
                        end_time = time()
                        key_meta=[self.keypoint_data_x,self.keypoint_data_y,self.keypoint_likelihood]

                        with h5py.File(os.path.join(key_point_meta_path,"meta_key.h5"), 'w') as f:
                            data_group=f.create_group("Facelab")
                            for i, tensor in zip(['x','y','p'],key_meta):
                                arr_um=[]
                                for tensor0 in tensor:
                                    arr_um.extend([tensor0.cpu().numpy()])
                                # print(arr_um)
                                data_group.create_dataset(i, data=arr_um)
                        print("key_point_meta_path",key_point_meta_path)
                        self.parent.slider_frame.setDisabled(False)
                        self.parent.temp_x, self.parent.temp_y, self.parent.temp_like = keypoint_check(key_point_meta_path)
                        # np.save(os.path.join(r"C:\LocalExpData\blinking2\2025-04-22_1\concatenated", "meta_key.npy"),np.array(key_meta, dtype=object),allow_pickle=True)
                        video_info.get_frame_number(self.parent, self.video_path, 0,tem_x=self.parent.temp_x, tem_y=self.parent.temp_y, tem_like=self.parent.temp_like)
                        self.log_signal.emit(f"Duration of the process: {end_time - start_time:.2f} seconds")
                        self.log_signal.emit(f"Size of keypoint data is: {len(self.keypoint_data_x)}" )
                        print("\n", f"Duration of the process: {end_time - start_time:.2f} seconds")
                        print("Size of keypoint data is", len(self.keypoint_data_x))

        elif self.parent.blink.isChecked():
            self.chunk = []
            start_time = time()
            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret = True
            while ret:
                for _ in tqdm(range(self.overal_frame + 1), desc="frames"):
                    ret, frame0 = cap.read()
                    if ret:
                        img = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
                        img=cv2.resize(img,(256,256))
                        if len(self.chunk) < self.chunk_size:
                            self.chunk.append(img)
                        if len(self.chunk) == self.chunk_size:
                            for ind_b, val_b in enumerate(self.blink_arr):
                                self.parent.ssim_arr["blink"][ind_b].extend(np.sum(255 - np.array(self.chunk)[:, val_b[1], val_b[0]], axis=-1))
                            self.log_signal.emit(f"{len(self.parent.ssim_arr['blink'][0])}")
                            self.chunk = []
                    elif not ret and len(self.chunk) != 0:
                        for ind_b, val_b in enumerate(self.blink_arr):
                            self.parent.ssim_arr["blink"][ind_b].extend(np.sum(255 - np.array(self.chunk)[:, val_b[1], val_b[0]], axis=-1))
                        self.log_signal.emit("FINISHED!")
                        self.parent.slider_frame.setDisabled(False)
                        self.chunk = []
            else:
                self.parent.save_roi_info(self.parent.ROIs)

                ssim_filename = Path(self.video_path).stem
                np.save(os.path.join(split(self.video_path)[0], f"{ssim_filename}_extracted_ssim.npy"),self.parent.ssim_arr)
                self.parent.ssim_plot_loaded = True
                self.parent.ssim_plot(os.path.join(split(self.video_path)[0], f"{ssim_filename}_extracted_ssim.npy"))
                self.parent.extract_ssim.setDisabled(False)
                end_time = time()
                print(f"Duration of the process: {end_time - start_time:.2f} seconds")










