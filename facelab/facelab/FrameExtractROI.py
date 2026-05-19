import cv2
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtCore
import pyqtgraph as pg



class frame_extract_roi_class:
    def __init__(self,fr_roi_pos=None,fr_roi_size=None,fr_roi_shape=None,fr_roi_color=None,parent=None):
        self.fr_iROI = parent.fr_nROIs
        view = parent.video_view.viewRange()
        imx = (view[0][1] + view[0][0]) / 2
        imy = (view[1][1] + view[1][0]) / 2
        dx = (view[0][1] - view[0][0]) / 4
        dy = (view[1][1] - view[1][0]) / 4
        dx = np.minimum(dx, parent.fr_image.shape[1] * 0.4)
        dy = np.minimum(dy, parent.fr_image.shape[0] * 0.4)
        imx = imx - dx / 3
        imy = imy - dy / 3

        if fr_roi_pos:
            imx=fr_roi_pos[1]
            imy = fr_roi_pos[0]
            dx=fr_roi_size[1]
            dy = fr_roi_size[0]
            self.roi_shape = fr_roi_shape
            self.color=fr_roi_color
        else:
            self.roi_shape = parent.fr_combo_box.currentText()
            self.color = np.maximum(0, np.minimum(255, np.random.randint(0, 255, 3)))

        self.draw(parent,imy,imx,dy,dx)
        self.ROI.sigRegionChangeFinished.connect(lambda: self.position(parent))
        self.ROI.sigClicked.connect(lambda : self.position(parent))
        self.ROI.sigRemoveRequested.connect(lambda : self.remove(parent))


    def draw(self,parent,imy,imx,dy,dx):
        print("One ROI added")
        self.draw_flag=True
        roipen=pg.mkPen(self.color,width=3,style=QtCore.Qt.SolidLine)

        if self.roi_shape=="Rectangular" or self.roi_shape=="Mask out":
           self.ROI = pg.RectROI([imx, imy], [dx, dy], pen=roipen, removable=True, movable=True)
        else:self.ROI = pg.EllipseROI([imx, imy], [dx, dy], pen=roipen, removable=True, movable=True)
        self.ROI.handleSize=8
        self.ROI.handlePen=roipen
        self.ROI.addScaleHandle([1,0.5],[0,0.5])
        self.ROI.addScaleHandle([0.5, 0], [0.5, 1])
        self.ROI.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        parent.video_view.addItem(self.ROI)


    def position(self,parent):
        if parent.fr_deleted_roi and self.fr_iROI > parent.fr_deleted_roi[0]:
            self.fr_iROI = self.fr_iROI - 1
            parent.fr_deleted_roi = [self.fr_iROI]
        br = self.ROI.boundingRect()
        pos0 = self.ROI.getSceneHandlePositions()
        pos = parent.video_view.mapSceneToView(pos0[0][1])
        if parent.fr_ROIs[self.fr_iROI].roi_shape == "Rectangular" or parent.fr_ROIs[self.fr_iROI].roi_shape == "Mask out":
            posy = pos.y()
        else:posy=pos.y()+br.height() / 2
        posx=pos.x()
        sizex, sizey = self.ROI.size()
        xrange = (np.arange(-1 * int(sizex), 1) + int(posx)).astype(np.int32)
        yrange = (np.arange(-1 * int(sizey), 1) + int(posy)).astype(np.int32)
        self.pos = posy, posx, posy + sizey, posx + sizex
        ellipse = np.zeros((yrange.size, xrange.size), "bool")
        x, y = np.meshgrid(np.arange(0, xrange.size, 1), np.arange(0, yrange.size, 1))
        ellipse = ((y - br.center().y()) ** 2 / (br.height() / 2) ** 2 + (x - br.center().x()) ** 2 / (
                    br.width() / 2) ** 2) <= 1
        # (x-h)^2/a^2+(y-k)^2/b^2 <= 1
        self.ellipse = ellipse[:, np.logical_and(xrange >= 0, xrange < parent.fr_image.shape[1])]  #
        xrange = xrange[np.logical_and(xrange >= 0, xrange < parent.fr_image.shape[1])]  # xrange=[-1,0,1,2,3] become [0,1,2,3]

        self.ellipse = ellipse[np.logical_and(yrange >= 0, yrange < parent.fr_image.shape[0]), :]
        yrange = yrange[np.logical_and(yrange >= 0, yrange < parent.fr_image.shape[0])]
        # print("xrange[xx], yrange[yy]",xrange, yrange)
        if parent.fr_ROIs[self.fr_iROI].roi_shape == "Rectangular" or parent.fr_ROIs[self.fr_iROI].roi_shape == "Mask out":

            yy, xx = np.where(self.ellipse != None)
        else:
            yy, xx = np.where(self.ellipse == True)

        self.new_x, self.new_y = (xrange[xx], yrange[yy])

        if self.draw_flag:
           parent.fr_pos_holder.append([self.new_x,self.new_y,parent.fr_ROIs[self.fr_iROI].roi_shape])
        elif len(parent.fr_pos_holder)!=0:
            parent.fr_pos_holder[self.fr_iROI] = [self.new_x,self.new_y,parent.fr_ROIs[self.fr_iROI].roi_shape]

        # self.frame_mask(parent)
        parent.frame_mask()
        self.draw_flag = False

    def remove(self,parent):
        parent.video_view.removeItem(self.ROI)
        del parent.fr_pos_holder[self.fr_iROI]
        del parent.fr_ROIs[self.fr_iROI]
        parent.fr_deleted_roi = [self.fr_iROI]

        parent.fr_nROIs -= 1
        print("fr_remove")
        for i in range(len(parent.fr_ROIs)):
             parent.fr_ROIs[i].position(parent)
        parent.fr_deleted_roi=[]
        if not parent.fr_ROIs:
            parent.roi_item.clear()


    # def frame_mask(self,parent):
    #     self.masked_image0 = parent.fr_image.copy()
    #     self.masked_out0 = np.zeros_like(parent.fr_image)
    #     for i in range(len(parent.fr_pos_holder)):
    #         if parent.fr_pos_holder[i][2] == "Mask out":
    #             self.cut_image = parent.fr_image[parent.fr_pos_holder[i][1][0]:parent.fr_pos_holder[i][1][-1],parent.fr_pos_holder[i][0][0]:parent.fr_pos_holder[i][0][-1]]
    #             # parent.save_masked_image = self.cut_image
    #             mask_out_temp = [[parent.fr_pos_holder[i][1][ii], parent.fr_pos_holder[i][0][ii]] for ii in range(len(parent.fr_pos_holder[i][0]))]
    #             for kk in mask_out_temp:
    #                 self.masked_out0[kk[0], kk[1]] = parent.fr_image[kk[0], kk[1]]
    #             self.masked_image0 = self.masked_out0
    #
    #     for i in range(len(parent.fr_pos_holder)):
    #         if parent.fr_pos_holder[i][2] != "Mask out":
    #             self.masked_image0[parent.fr_pos_holder[i][1], parent.fr_pos_holder[i][0]] = [0]
    #
    #     # parent.save_masked_image = self.masked_image0
    #
    #     # self.masked_image = np.transpose(self.masked_image0, (1, 0, 2))
    #     parent.roi_item.setImage(self.masked_image0)
