import numpy as np
import torch
import cv2
from PyQt5.QtCore import QThread, pyqtSignal

import facelab.keypoint_fine_tuning as keypoint_fine_tuning
import facelab.keypoint_prediction as keypoint_prediction
from facelab.keypoint_model_training import Keypoint_Class
from facelab.video_info import get_frame_number


class finetune_thread(QThread):
    log_signal_finetune=pyqtSignal(str)
    def __init__(self,parent=None,model=None):
        super().__init__()
        self.parent = parent
        self.rand_frames = np.random.choice(self.parent.overall_frame, 250, replace=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            self.model = Keypoint_Class().to(self.device)
            self.selected_model = self.parent.model_name[self.parent.initial_model_combo.currentIndex()]
            self.model.load_state_dict(torch.load(self.selected_model, map_location=self.device)["model_state_dict"])
        else:
            self.model = model
        self.frame_chunck = []




    def run(self):
        cap = cv2.VideoCapture(self.parent.video_path[0])
        for r_f_ind, r_f_val in enumerate(self.rand_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, r_f_val)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (256, 256))
            self.frame_chunck.append(frame)
            # if r_f_ind % 50 == 0:
            #     selected_frame=int(self.parent.overall_frame*(r_f_ind/250))
            #     get_frame_number(self.parent, self.parent.video_path[0], selected_frame)
        x_key, y_key, likelihood_key = keypoint_prediction.predict_net(self.frame_chunck, self.model, self.device,log=self.log_signal_finetune)


        x_key_list = [x_k.cpu().numpy() for x_k in x_key]
        y_key_list = [y_k.cpu().numpy() for y_k in y_key]
        likelihood_key_list = [l_k.cpu().numpy() for l_k in likelihood_key]
        avg_likelihood_key = np.mean(likelihood_key_list, axis=-1)
        perc_likelihood = np.nanpercentile(avg_likelihood_key, 95)
        difficult_frames = np.where(avg_likelihood_key <= perc_likelihood)[0]
        easy_frames = np.where(avg_likelihood_key > perc_likelihood)[0]
        frame_set = [*difficult_frames, *easy_frames]
        frame_set = [fr for fr in frame_set if fr not in self.parent.frame_set_arr][:self.parent.frame_val]
        likelihood_key = [likelihood_key_list[f_vh] for f_vh in frame_set[:self.parent.frame_val]]
        x_key = [x_key_list[f_vh] for f_vh in frame_set[:self.parent.frame_val]]
        y_key = [y_key_list[f_vh] for f_vh in frame_set[:self.parent.frame_val]]
        frame_chunk = [self.frame_chunck[f_vh] for f_vh in frame_set[:self.parent.frame_val]]

        self.parent.x_key_arr = [*x_key, *self.parent.x_key_arr]
        self.parent.y_key_arr = [*y_key, *self.parent.y_key_arr]
        self.parent.like_key_arr = [*likelihood_key, *self.parent.like_key_arr]
        self.parent.frame_set_arr = [*frame_set, *self.parent.frame_set_arr]
        self.parent.frame_data_arr = [*frame_chunk, *self.parent.frame_data_arr]
        print("self.parent.x_key_arr",len(self.parent.x_key_arr))
        print(" self.parent.like_key_arr",len( self.parent.like_key_arr))
        self.log_signal_finetune.emit("finished")


        # return x_key_final, y_key_final, likelihood_key_final, frame_set[:self.parent.frame_val], frame_chunk






class finetune_prediction:
    def __init__(self,parent=None,model=None):
        self.parent=parent
        self.rand_frames=np.random.choice(self.parent.overall_frame,250,replace=False)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            self.model= Keypoint_Class().to(self.device)
            self.selected_model=self.parent.model_name[self.parent.initial_model_combo.currentIndex()]
            self.model.load_state_dict(torch.load(self.selected_model, map_location=self.device)["model_state_dict"])
        else:
            self.model=model
        self.frame_chunck=[]

    def finetune_data(self):
        cap=cv2.VideoCapture(self.parent.video_path[0])
        for r_f_ind,r_f_val in enumerate(self.rand_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES,r_f_val)
            ret,frame=cap.read()
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            frame=cv2.resize(frame,(256,256))
            self.frame_chunck.append(frame)
            if r_f_ind %50==0:
                selected_frame = int(self.parent.overall_frame * (r_f_ind / 250))
                print(selected_frame)
                get_frame_number(self.parent,self.parent.video_path[0],selected_frame)

        x_key, y_key, likelihood_key = keypoint_prediction.predict_net(self.frame_chunck, self.model, self.device)

        x_key_list = [x_k.cpu().numpy() for x_k in x_key]
        y_key_list = [y_k.cpu().numpy() for y_k in y_key]
        likelihood_key_list=[l_k.cpu().numpy() for l_k in likelihood_key]
        avg_likelihood_key=np.mean(likelihood_key_list,axis=-1)
        perc_likelihood=np.nanpercentile(avg_likelihood_key,95)
        difficult_frames=np.where(avg_likelihood_key<=perc_likelihood)[0]
        easy_frames = np.where(avg_likelihood_key > perc_likelihood)[0]
        frame_set=[*difficult_frames,*easy_frames]
        frame_set=[fr for fr in frame_set if fr not in self.parent.frame_set_arr]
        likelihood_key_final=[likelihood_key_list[f_vh] for f_vh in frame_set[:self.parent.frame_val]]
        x_key_final = [x_key_list[f_vh] for f_vh in frame_set[:self.parent.frame_val]]
        y_key_final = [y_key_list[f_vh] for f_vh in frame_set[:self.parent.frame_val]]
        frame_chunk = [self.frame_chunck[f_vh] for f_vh in frame_set[:self.parent.frame_val]]


        return x_key_final,y_key_final,likelihood_key_final,frame_set[:self.parent.frame_val],frame_chunk








