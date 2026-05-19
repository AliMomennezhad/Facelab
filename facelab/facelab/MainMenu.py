import glob
import os
import sys
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import QAction, QFileDialog, QMessageBox
import h5py

import facelab.finetune_preparation as finetune_preparation
import facelab.frame_extraction as frame_extraction
import facelab.keypoint_fine_tuning as keypoint_fine_tuning
import facelab.keypoint_training as keypoint_training

import facelab.video_info as video_info
def mainmenu(parent):
    main_menu=parent.menuBar()
    file_menu = main_menu.addMenu("File")
    parent.keypoint_training_menu = main_menu.addMenu("Keypoint Training")
    parent.frame_extraction_menu=main_menu.addMenu("Frame extraction")
    frame_extraction_action=QAction("Frame Extraction",parent)
    load_roi_menu = main_menu.addMenu("Load ROIs")
    load_action = QAction("Load", parent)
    keypoint_training_action=QAction("Keypoint training",parent)
    keypoint_fine_tuning_action = QAction("Keypoint FineTuning", parent)
    load_roi_action = QAction("Load Existing ROIs", parent)
    load_action.triggered.connect(lambda:load_video(parent,drag=False)) #if we don't pass arguments like(parent) to load_video we don't need lambda
    load_roi_action.triggered.connect(lambda: load_roi(parent))
    parent.keypoint_training_menu.addAction(keypoint_training_action)
    parent.keypoint_training_menu.addAction(keypoint_fine_tuning_action)
    keypoint_training_action.triggered.connect(lambda:keypoint_function(parent))
    keypoint_fine_tuning_action.triggered.connect(lambda: keypoint_fine_tuning_function(parent))
    frame_extraction_action.triggered.connect(lambda: frame_extraction_function(parent))
    file_menu.addAction(load_action)
    load_roi_menu.addAction(load_roi_action)
    parent.frame_extraction_menu.addAction(frame_extraction_action)
    parent.frame_extraction_menu.setEnabled(False)
    parent.keypoint_training_menu.setEnabled(False)

def keypoint_function(parent):
    keypoint_training.DraggablePointDialog(parent)

def keypoint_fine_tuning_function(parent):
    finetune_preparation.finetune_prep(parent)
    # keypoint_fine_tuning.DraggablePointDialog(parent)
def frame_extraction_function(parent):
    if parent.frame_proto_loaded:
       frame_extraction.Frame_Extraction(parent)
    else:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setText("Choose a file containing experiment info first")
        msg.setWindowTitle("Error!")
        # msg.setStandardButtons(QMessageBox.StandaradButton.Ok)
        msg.exec_()

def load_video(parent,drag,video=None):
    if drag==False:
        file_path, _ = QFileDialog.getOpenFileName(
            parent,
            "Open Video File",
            "C:\LocalExpData",
            "Video Files (*.mp4 *.avi *.mov *.mkv);"
        )
    else:file_path=video
    if file_path:
        parent.keypoint_training_menu.setEnabled(True)
        parent.frame_extraction_menu.setEnabled(True)
        parent.add_roi_btn.setDisabled(False)
        parent.play_btn.setDisabled(False)
        parent.pause_btn.setDisabled(False)
        parent.blink.setEnabled(True)
        parent.ssim_extract.setEnabled(True)
        parent.keypoint_check.setEnabled(True)
        # parent.train_check.setEnabled(True)
        path = Path(file_path)
        parent_file = path.parent
        stem_file = path.stem
        parent.video_path = [file_path]
        parent.setWindowTitle(f"FaceLab- {parent.video_path[0]}")
        parent.scatter_item.clear()
        parent.temp_x,parent.temp_y,parent.temp_like=keypoint_check(parent_file)
        print(f"Video file selected: {file_path}")
        parent.log_display.append(f"\n{file_path}")
        parent.svd_plot_window_plot.clear()
        if parent.ROIs:
            parent.roi_pg.clear()
            parent.save_masked_image = []
            for roi in parent.ROIs:
                parent.image_viewbox.removeItem(roi.ROI)
        parent.ROIs = []
        parent.nROIs = 0
        parent.pos_holder = []

        overall_frame=str(video_info.get_frame_number(parent,file_path,selected_frame=0,tem_x=parent.temp_x,tem_y=parent.temp_y,tem_like=parent.temp_like))
        parent.videoframes.setText(f"/ {overall_frame}")
        parent.slider_frame.setDisabled(False)
        parent.slider_frame.setValue(0)
        parent.slider_frame.setMaximum(int(overall_frame))
        parent.frame_proto_loaded = False

        # npy_files = glob.glob(os.path.join(parent_file, "*.npy"))
        ##clearing the viewbox before adding loading new video


        create_path=os.path.join(parent_file, f"{stem_file}_extracted_ssim.npy")
        if os.path.exists(create_path):
            parent.ssim_plot_loaded=True
            parent.ssim_plot(create_path)

def load_roi(parent):
    roi_path, _ = QFileDialog.getOpenFileName(
        parent,
        "Open File Containing ROI)",
        "C:\LocalExpData",
        "ROI Files (*.npy);"
    )
    if roi_path:
        try:
            roi_file=np.load(roi_path, allow_pickle=True).item()
            parent.load_roi(roi_file)
        except Exception as e:
            msg=QMessageBox()
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setText("No file Containing ROI")
            msg.setWindowTitle("Error!")
            # msg.setStandardButtons(QMessageBox.StandaradButton.Ok)
            msg.exec_()


def keypoint_check(video_dir):
    key_path=os.path.join(video_dir,"meta_key.h5")
    if os.path.exists(key_path):
        with h5py.File(key_path, 'r') as f:

            # print(f.keys())
            temp_x=f['Facelab']["x"][:]
            temp_y = f['Facelab']["y"][:]
            temp_like=f['Facelab']["p"][:]
        return temp_x,temp_y,temp_like
    else:return None,None,None








#
# runface_analysis=QAction("&action",parent)
# parent.addAction(runface_analysis)
#
# file_menu=main_menu.addMenu("&File")
# file_menu.addAction(runface_analysis)
#
# nice_menu=main_menu.addMenu("&Nice")
#
# # Add "File" menu
# file_menu = menubar.addMenu("File")
#
# # Add "Load" action
# load_action = QAction("Load", self)
# load_action.triggered.connect(self.load_file)
# file_menu.addAction(load_action)