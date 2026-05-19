import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtGui import QIntValidator
import pyqtgraph as pg
from facelab.class_for_ROI import class_for_roi
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

# temmp=np.load(r"C:\LocalExpData\blinking2\2025-04-22_1\concatenated\meta_key.npy",allow_pickle=True)
# with h5py.File(r"C:\LocalExpData\blinking 26\2025-05-16_1\concatenated\meta_key.h5", 'r') as f:
#     temp_x=f["0"][:]
#     temp_y = f["1"][:]
#     temp_like=f["2"][:]
#     paw_like_arr=[paw_like[0] for paw_like in temp_like]
#     min_likelihood_paw=np.where(np.array(paw_like_arr)<=0.51)[0]

def get_frame_number(parent,video,selected_frame,tem_x=None,tem_y=None,tem_like=None):
    cap=cv2.VideoCapture(video)
    selected_frame=min(selected_frame,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
    ret, frame0 = cap.read()
    if ret:
        parent.image=cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)
        parent.image=cv2.resize(parent.image,(256,256))
        parent.image1=np.transpose(parent.image, (1, 0))
        parent.image_item.setImage(parent.image1)
        if isinstance(tem_x, np.ndarray):
            selected_temp_x=tem_x[selected_frame]
            selected_temp_y = tem_y[selected_frame]
            selected_likelihood=tem_like[selected_frame]
            parent.scatter_item.setData(pos=[[selected_temp_y[0],selected_temp_x[0]],[selected_temp_y[1],selected_temp_x[1]],[selected_temp_y[2],selected_temp_x[2]]],data=[f'paw-{selected_likelihood[0]}',f'point1-{selected_likelihood[1]}',f'point2-{selected_likelihood[2]}'],hoverable=True,hoverSymbol="x",pxMode=True,hoverBrush="r",symbolPen=pg.mkPen(color=(255, 255, 255)), size=6, symbol='o',brush=[QtGui.QColor('DarkOrange'),QtGui.QColor('blue'),QtGui.QColor('yellow')])
        parent.frame_num.setReadOnly(False)
        parent.frame_num.setText(str(selected_frame))
        validator = QIntValidator(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        parent.frame_num.setValidator(validator)
        parent.overall_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if parent.ROIs:
            class_for_roi.mask_image(parent,parent,parent.pos_holder,parent.image)


    return parent.overall_frame

