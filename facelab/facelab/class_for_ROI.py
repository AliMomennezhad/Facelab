import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtGui import QImage

class class_for_roi:
    def __init__(self,roi_pos=None,roi_size=None,roi_shape=None,roi_color=None,parent=None):

        self.iROI=parent.nROIs
        view=parent.image_viewbox.viewRange()
        imx=(view[0][1]+view[0][0])/2
        imy=(view[1][1]+view[1][0])/2
        dx=(view[0][1]-view[0][0])/4
        dy = (view[1][1] - view[1][0]) / 4
        dx = np.minimum(dx, parent.image.shape[1]*0.4)
        dy = np.minimum(dy, parent.image.shape[0]* 0.4)
        imx=imx-dx/3
        imy=imy-dy/3

        if roi_pos:
            imx=roi_pos[1]
            imy = roi_pos[0]
            dx=roi_size[1]
            dy = roi_size[0]
            self.roi_shape = roi_shape
            self.color=roi_color
        else:
            self.roi_shape = parent.combo_box.currentText()
            self.color = np.maximum(0, np.minimum(255, np.random.randint(0, 255, 3)))

        self.draw(parent,imy,imx,dy,dx)
        self.ROI.sigRegionChangeFinished.connect(lambda: self.position(parent))
        self.ROI.sigClicked.connect(lambda : self.position(parent))
        self.ROI.sigRemoveRequested.connect(lambda : self.remove(parent))


    def draw(self,parent,imy,imx,dy,dx):
        self.draw_flag=True

        roipen=pg.mkPen(self.color,width=3,style=QtCore.Qt.SolidLine)
        # print("imy,imx,dy,dx",imy,imx,dy,dx)
        # self.ROI=pg.RectROI([imx,imy],[dx,dy],pen=roipen,removable=True,movable=True)

        if self.roi_shape=="Rectangular" or self.roi_shape=="Mask out":
           self.ROI = pg.RectROI([imx, imy], [dx, dy], pen=roipen, removable=True, movable=True)
        else:self.ROI = pg.EllipseROI([imx, imy], [dx, dy], pen=roipen, removable=True, movable=True)
        self.ROI.handleSize=8
        self.ROI.handlePen=roipen
        self.ROI.addScaleHandle([1,0.5],[0,0.5])
        self.ROI.addScaleHandle([0.5, 0], [0.5, 1])
        self.ROI.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        parent.image_viewbox.addItem(self.ROI)

        # self.position(parent)

    def position(self,parent):

        if parent.deleted_roi and self.iROI > parent.deleted_roi[0]:
            self.iROI = self.iROI - 1
            parent.deleted_roi = [self.iROI]

        br = self.ROI.boundingRect()
        pos0=self.ROI.getSceneHandlePositions()
        pos=parent.image_viewbox.mapSceneToView(pos0[0][1])
        if parent.ROIs[self.iROI].roi_shape == "Rectangular" or parent.ROIs[self.iROI].roi_shape == "Mask out":
            posy = pos.y()
        else:posy=pos.y()+br.height() / 2
        posx=pos.x()
        sizex,sizey=self.ROI.size()
        xrange = (np.arange(-1 * int(sizex), 1) + int(posx)).astype(np.int32)
        yrange = (np.arange(-1 * int(sizey), 1) + int(posy)).astype(np.int32)
        self.pos=posy,posx,posy+sizey,posx+sizex

        ellipse = np.zeros((yrange.size, xrange.size), "bool")
        x, y = np.meshgrid(np.arange(0, xrange.size, 1), np.arange(0, yrange.size, 1))
        ellipse = ((y - br.center().y()) ** 2 / (br.height() / 2) ** 2 + (x - br.center().x()) ** 2 / (br.width() / 2) ** 2) <= 1
        #(x-h)^2/a^2+(y-k)^2/b^2 <= 1
        self.ellipse = ellipse[:, np.logical_and(xrange >= 0, xrange < parent.image.shape[1])] #
        xrange = xrange[np.logical_and(xrange >= 0, xrange < parent.image.shape[1])] #xrange=[-1,0,1,2,3] become [0,1,2,3]
        self.ellipse = ellipse[np.logical_and(yrange >= 0, yrange < parent.image.shape[0]), :]
        yrange = yrange[np.logical_and(yrange >= 0, yrange < parent.image.shape[0])]
        # print("xrange[xx], yrange[yy]",xrange, yrange)
        if parent.ROIs[self.iROI].roi_shape == "Rectangular" or parent.ROIs[self.iROI].roi_shape == "Mask out":
            yy, xx = np.where(self.ellipse !=None)
        else:yy,xx=np.where(self.ellipse ==True)
        # print(self.ellipse)

        self.new_x, self.new_y= (xrange[xx], yrange[yy])

        # newimage=parent.image.copy()
        # newimage[yrange[yy],xrange[xx]] = [0, 0, 0]
        #
        # plt.figure(figsize=(8, 6))
        # plt.imshow(newimage)
        # plt.title('Masked Pixels')
        # plt.axis('off')
        # plt.show()

        #
        # plt.imshow(parent.image)
        # plt.scatter(xrange[xx], yrange[yy], c='red', s=1)  # s is size of the spot
        # plt.axis('off')
        # plt.title("Image with Red Spots")
        # plt.show()

        if self.draw_flag:
           # parent.pos_holder.append([self.x1, self.y1, self.x2, self.y2])
           parent.pos_holder.append([self.new_x,self.new_y,parent.ROIs[self.iROI].roi_shape])
        elif len(parent.pos_holder)!=0:
            # parent.pos_holder[self.iROI] = [self.x1, self.y1, self.x2, self.y2]
            parent.pos_holder[self.iROI] = [self.new_x,self.new_y,parent.ROIs[self.iROI].roi_shape]

        # print("parent.pos_holder",parent.pos_holder)
        self.mask_image(parent, parent.pos_holder, parent.image)
        self.draw_flag=False
        # print("parent.pos_holder position", parent.pos_holder)


    def remove(self, parent):
        parent.log_display.append("One ROI removed")
        parent.image_viewbox.removeItem(self.ROI)
        del parent.pos_holder[self.iROI]
        del parent.ROIs[self.iROI]
        parent.deleted_roi=[self.iROI]

        parent.nROIs -=1
        print("remove")
        if len(parent.pos_holder) == 0:
            self.masked_image = np.transpose(parent.image, (1, 0))
            parent.roi_pg.setImage(self.masked_image)
        else:

            self.mask_image(parent,parent.pos_holder,parent.image)

        for i in range(len(parent.ROIs)):
             parent.ROIs[i].position(parent)
        parent.deleted_roi=[]
        if not parent.ROIs:
            parent.roi_pg.clear()
            parent.save_masked_image =[]


    def mask_image(self,parent,pos_hold,orig_image):

        self.masked_image0=orig_image.copy()
        self.masked_out0 = np.zeros_like(orig_image)
        for i in range(len(pos_hold)):
            if pos_hold[i][2]=="Mask out":
                self.cut_image=orig_image[pos_hold[i][1][0]:pos_hold[i][1][-1],pos_hold[i][0][0]:pos_hold[i][0][-1]]
                parent.save_masked_image = self.cut_image
                print(np.shape(parent.save_masked_image))
                mask_out_temp = [[pos_hold[i][1][ii], pos_hold[i][0][ii]] for ii in range(len(pos_hold[i][0]))]
                for kk in mask_out_temp:
                    self.masked_out0[kk[0], kk[1]] = orig_image[kk[0], kk[1]]
                self.masked_image0=self.masked_out0

        for i in range(len(pos_hold)):
            if pos_hold[i][2] != "Mask out":
                self.masked_image0[pos_hold[i][1], pos_hold[i][0]] = [0]

        # parent.save_masked_image = self.masked_image0

        self.masked_image = np.transpose(self.masked_image0, (1, 0))
        parent.roi_pg.setImage(self.masked_image)

        # mask = np.ones((orig_image.shape[0], orig_image.shape[1]), dtype="uint8") * 255
        # for i in range(len(pos_hold)):
        #     cv2.rectangle(mask, (pos_hold[i][0], pos_hold[i][1]),(pos_hold[i][2], pos_hold[i][3]), 0,-1)
        #
        # self.masked_image0 = cv2.bitwise_and(orig_image, orig_image, mask=mask)
        # self.masked_image = np.transpose(self.masked_image0, (1, 0, 2))
        # parent.roi_pg.setImage(self.masked_image)







