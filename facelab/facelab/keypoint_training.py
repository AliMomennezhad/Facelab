import math
import os
import sys
from os import path

import numpy as np
import pyqtgraph as pg

from PyQt5.QtCore import Qt, QSize, QCoreApplication, QTimer
import cv2
from PyQt5.QtGui import QIcon, QGuiApplication
from PyQt5.QtWidgets import QApplication, QMenuBar, QAction, QFileDialog, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, \
    QMessageBox
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

import facelab.finetune_preparation as finetune_preparation
import facelab.Progress_bar_extraction as Progress_bar_extraction


class DraggablePointItem(pg.ScatterPlotItem):
    def __init__(self, pos,index,mydata,color=None,parent=None):
        super().__init__()
        self.position = np.array(pos)  # Store the initial position as an attribute
        self.color = color
        self.dddata=mydata

        self.setData(pos=[self.position],data=self.dddata,hoverable=True,hoverSymbol="x",pxMode=True,hoverBrush="r",symbolPen=pg.mkPen(color=(255, 255, 255)), size=10, symbol='o', brush=self.color)
        self.dragging = False  # Track dragging state
        self.index=index
        self.parent=parent

        # Enable focus to capture key events and hover events
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable)  # Enable focus for key events
        self.setAcceptHoverEvents(True)  # Enable hover events for this item
        # self.setAcceptMouseEvents(True)  # Enable mouse events for this item
        self.setFocus()

    def hoverEvent(self, event):
        """Handle hover events."""
        super().hoverEvent(event)  # Call the base class method to allow default behavior


    def mouseClickEvent(self, event):
        """Handle mouse click event on the point."""

        mouse_pos = event.pos()
        dist = np.linalg.norm(self.position - np.array([mouse_pos.x(), mouse_pos.y()]))

        if dist < 10:
            self.parent.selected_point=self.index


    def mouseDragEvent(self, event):

        if event.button() == Qt.LeftButton:



            if event.isStart():  # Start dragging
                mouse_pos = event.pos()
                dist = np.linalg.norm(self.position - np.array([mouse_pos.x(), mouse_pos.y()]))
                if dist < 10:  # 10-pixel threshold for clicking on the point

                    self.dragging = True
                    event.accept()
                else:
                    event.ignore()
            elif event.isFinish():  # Finish dragging

                self.dragging = False
            elif self.dragging:  # Continue dragging
                # Update point position while dragging

                new_pos = event.pos()
                # self.position = (np.array([new_pos.x(), new_pos.y()])).astype(int)
                self.position = (np.array([new_pos.x(), new_pos.y()])) # lets try non integer
                self.parent.poss=self.position
                self.parent.keypoint_coord[self.parent.current_frame][self.index]=self.position
                # print(self.parent.keypoint_coord)
                self.setData(pos=[self.position],data=self.dddata)  # Update the displayed position
                event.accept()
            else:
                event.ignore()


class DraggablePointDialog(QtWidgets.QDialog):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.Window |  # Basic window with title bar
            Qt.WindowMinimizeButtonHint |  # Enable minimize button
            Qt.WindowCloseButtonHint  # Enable close button
        )

        self.setWindowTitle("Keypoint")
        # Create a QLabel to display the image
        self.setGeometry(500, 100, 1000, 800)
        app_icon = QIcon()
        icon_path = r'keypoint_icon.png'
        app_icon.addFile(icon_path, QSize(16, 16))
        app_icon.addFile(icon_path, QSize(24, 24))
        app_icon.addFile(icon_path, QSize(32, 32))
        app_icon.addFile(icon_path, QSize(48, 48))
        app_icon.addFile(icon_path, QSize(96, 96))
        app_icon.addFile(icon_path, QSize(256, 256))
        QApplication.setWindowIcon(app_icon)
        QCoreApplication.setApplicationName("Keypoint GUI")

        centerPoint = QGuiApplication.primaryScreen().availableGeometry().center()
        qtRectangle = self.frameGeometry()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())
        pg.setConfigOptions(imageAxisOrder='row-major')
        # Initialize the layout
        center_layout = QVBoxLayout()
        self.setLayout(center_layout)
        self.win = pg.GraphicsLayoutWidget()
        center_layout.addWidget(self.win)
        self.next_timer=QTimer()
        self.prev_timer = QTimer()
        self.next_button=QPushButton(self)
        self.next_button.setText("Next Frame")
        self.next_button.setDisabled(True)

        self.previous_button = QPushButton(self)
        self.previous_button.setText("Previous Frame")
        self.previous_button.setDisabled(True)

        self.save_button = QPushButton(self)
        self.save_button.setText("SAVE")
        self.save_button.setDisabled(True)

        self.save_train_button = QPushButton(self)
        self.save_train_button.setText("SAVE and Train")
        self.save_train_button.setDisabled(True)


        self.frame_label=QLabel(self)

        button_layer = QHBoxLayout()
        button_layer.addWidget(self.previous_button)
        button_layer.addWidget(self.next_button)
        button_layer.addWidget(self.save_button)
        button_layer.addWidget(self.save_train_button)
        button_layer.addWidget(self.frame_label)


        center_layout.addLayout(button_layer)
        self.next_timer.timeout.connect(self.next_frame)
        self.next_button.clicked.connect(self.next_frame)
        self.next_button.pressed.connect(lambda:self.start_slide_show("next"))
        self.next_button.released.connect(lambda:self.stop_slide_show("next"))
        self.prev_timer.timeout.connect(self.previous_frame)
        self.previous_button.pressed.connect(lambda:self.start_slide_show("prev"))
        self.previous_button.released.connect(lambda:self.stop_slide_show("prev"))
        self.previous_button.clicked.connect(self.previous_frame)
        self.save_button.clicked.connect(self.save_keypoint)
        self.save_train_button.clicked.connect(self.save_train_keypoint)
        self.video_view = self.win.addViewBox(lockAspect=True, invertY=True)
        # self.image = cv2.imread(r"C:\VideoAnalysis\dat006\2024-09-14_1\prototype\prototype_airpuff\dat006-09-14_1_2 (8486).jpg")
        # self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image_item = pg.ImageItem()  # Example image data
        self.video_view.addItem(self.image_item)
        self.points = []
        self.cap = None  # VideoCapture object will be created when video is loaded
        self.colors = [QtGui.QColor('DarkOrange'), QtGui.QColor('blue'), QtCore.Qt.yellow,
                       QtCore.Qt.magenta, QtCore.Qt.cyan, QtCore.Qt.darkRed, QtCore.Qt.darkGreen,
                       QtCore.Qt.darkBlue, QtCore.Qt.darkYellow, QtCore.Qt.darkMagenta, QtCore.Qt.darkCyan,
                       QtCore.Qt.gray, QtCore.Qt.darkGray, QtCore.Qt.lightGray]
        self.overall_frame=parent.overall_frame
        self.parent=parent
        # self.video_view.setMouseEnabled(x=True, y=True)
        self.video_path = parent.video_path[0]
        self.load_video()
        self.show()

    def load_video(self):

        # video_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        video_path=self.video_path
        if video_path:
            self.initial_load()

            # Release previous video if any
            if self.cap is not None:
                self.cap.release()

            # Load the new video
            self.cap = cv2.VideoCapture(video_path)
            # self.current_frame = 0  # Reset current frame index
            self.frame_label.setText(str(self.current_frame))
            self.setWindowTitle(f"Frame-by-Frame Video Viewer - {video_path}")
            self.show_frame()  # Show the first frame

            for point in self.points:
                self.video_view.addItem(point)

            # self.keypoint_info["current_frame"]=self.current_frame
            # self.keypoint_info["keypont_cord"] =
            if len(self.keypoint_coord) <= self.current_frame:
                self.keypoint_coord.append([self.points[0].position,self.points[1].position,self.points[2].position])
                self.frame_container.append(self.frame_gray)
            print("keypoint",len(self.keypoint_coord))
            print("frame",len(self.frame_container))
            self.next_button.setDisabled(False)
            self.previous_button.setDisabled(False)
            self.save_button.setDisabled(False)
            self.save_train_button.setDisabled(False)


    def initial_load(self):
        for i in range(len(self.points)):
            print(self.points[i])
            self.video_view.removeItem(self.points[i])
            # self.points.remove(self.points[i])  # Remove point from the points list
        if len(self.points) > 0:
            self.points = []
        # self.selected_point = []
        self.poss = []
        self.keypoint_frame_info = {}
        self.keypoint_coord = []
        self.frame_container = []
        self.current_frame = 0  # Track the current frame position
        print("load video was clicked")

        if path.exists(r"C:\VideoAnalysis\keypointdata\kepoint_data.npy"):
            self.keypoint_frame_info = np.load(r"C:\VideoAnalysis\keypointdata\kepoint_data.npy",allow_pickle=True).item()
            self.keypoint_coord = self.keypoint_frame_info["keypoint"]
            self.frame_container = self.keypoint_frame_info["frame"]
            self.current_frame=min(len(self.keypoint_coord)-1,self.overall_frame)
            print("keypoint", self.keypoint_coord)
            print("frame", len(self.frame_container))

            self.points.append(DraggablePointItem(pos=self.keypoint_coord[-1][0], index=0, mydata="paw",color=QtGui.QColor(self.colors[0]), parent=self))
            self.points.append(DraggablePointItem(pos=self.keypoint_coord[-1][1], index=1, mydata="point1",color=QtGui.QColor(self.colors[1]), parent=self))
            self.points.append(DraggablePointItem(pos=self.keypoint_coord[-1][2], index=2, mydata="point2",color=QtGui.QColor(self.colors[2]), parent=self))
            # self.points.append(DraggablePointItem(pos=self.keypoint_coord[0][3], index=3, mydata="spout",color=QtGui.QColor(self.colors[3]), parent=self))
            self.keypoint_coord = list(self.keypoint_coord)
            print(self.keypoint_coord)

        else:
            self.points.append(DraggablePointItem(pos=[30, 30], index=0, mydata="paw", color=QtGui.QColor(self.colors[0]),parent=self))
            self.points.append(DraggablePointItem(pos=[80, 80], index=1, mydata="point1", color=QtGui.QColor(self.colors[1]),parent=self))
            self.points.append(DraggablePointItem(pos=[30, 80], index=2, mydata="point2", color=QtGui.QColor(self.colors[2]),parent=self))
            # self.points.append(
            #     DraggablePointItem(pos=[70, 20], index=3, mydata="spout", color=QtGui.QColor(self.colors[3]),
            #                        parent=self))

        self.selected_point = None

    def show_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()

        if not ret:
            return

        # Convert the frame to RGB format
        self.frame_gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_gray=cv2.resize(self.frame_gray0,(256,256))
        # self.frame_gray=self.resize_with_padding(self.frame_gray0)



        # Set image data to ImageItem
        self.image_item.setImage(self.frame_gray)

    def resize_with_padding(self,myimage, target_height=256, target_width=256):
        print("what is the shape)",myimage.shape[:2])
        # Calculate padding for height and width
        original_height, original_width = myimage.shape[:2]
        pad_height = target_height - original_height
        pad_width = target_width - original_width

        # Compute padding values for top, bottom, left, right
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left

        # Apply padding
        padded_image = cv2.copyMakeBorder(myimage, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0 )  # value=0 pads with black; change as needed

        return padded_image


    def start_slide_show(self,next_prev):
        if next_prev=="next":
          self.next_timer.start(100)
        else:self.prev_timer.start(100)
    def stop_slide_show(self,next_prev):
        if next_prev == "next":
            self.next_timer.stop()
        else:
            self.prev_timer.stop()

    def next_frame(self):
        """Advance to the next frame."""
        if self.cap and self.cap.isOpened():
            # Move to the next frame and display it
            self.current_frame += 1
            if self.current_frame >= int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                self.current_frame = 0  # Loop back to the start
            self.show_frame()
            if self.current_frame != 0 and len(self.keypoint_coord)<=self.current_frame :
               self.keypoint_coord.append([self.points[0].position, self.points[1].position, self.points[2].position])
               self.frame_container.append(self.frame_gray)
        self.frame_label.setText(str(self.current_frame))
        self.update_key()
        print("keypoint",len(self.keypoint_coord))
        print("frame", len(self.frame_container))





    def previous_frame(self):
        """Go back to the previous frame."""
        if self.cap and self.cap.isOpened():
            # Move to the previous frame and display it
            self.current_frame -= 1
            if self.current_frame < 0:
                self.current_frame = 0  # Loop back to the end
            self.show_frame()
            self.frame_label.setText(str(self.current_frame))
            self.update_key()
            print("keypoint", len(self.keypoint_coord))
            print("frame", len(self.frame_container))



    def update_key(self):
        for i in range(len(self.points)):
            self.points[i].position = self.keypoint_coord[self.current_frame][i]
            self.points[i].setData(pos=[self.keypoint_coord[self.current_frame][i]], data=self.points[i].dddata)

    def save_train_keypoint(self):
        self.keypoint_frame_info["keypoint"] = self.keypoint_coord
        self.keypoint_frame_info["frame"] = self.frame_container
        keypoint_saving_path = "C:\VideoAnalysis\keypointdata"
        model_saving_path="C:\LocalExpData\models"
        self.destination_path=os.path.join(model_saving_path,'meta_model_training.pt')
        os.makedirs(keypoint_saving_path, exist_ok=True)

        np.save(os.path.join(keypoint_saving_path, "kepoint_data.npy"), self.keypoint_frame_info)
        self.fine_tune_instance = finetune_preparation.fine_tune_thread(keypoint_saving_path, 0.001,
                                                                        destination_path=self.destination_path,
                                                                        model_path=None, parent=self,epoch=500)

        self.fine_tune_instance.log_signal.connect(self.finetune_update)
        self.fine_tune_progess = Progress_bar_extraction.Progress_bar_extraction(self,gui=self,main_gui=self.parent)
        self.fine_tune_instance.start()

    def finetune_update(self,message):
        self.fine_tune_progess.update_progress(message,training=True,destination_path=self.destination_path)

    def save_keypoint(self):
        self.keypoint_frame_info["keypoint"]=self.keypoint_coord
        self.keypoint_frame_info["frame"]=self.frame_container
        keypoint_saving_path="C:\VideoAnalysis\keypointdata"
        os.makedirs(keypoint_saving_path,exist_ok=True)
        np.save(os.path.join(keypoint_saving_path,"kepoint_data.npy"),self.keypoint_frame_info)
        msg=QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText("Keypoints successfully saved!")
        msg.setWindowTitle("Keypoint Saving")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec_()

    def closeEvent(self, event):
        # Release video capture and stop timer on close
        if self.cap:
            self.cap.release()
        # self.timer.stop()
        event.accept()


    def keyPressEvent(self, event):
        """Handle key press events, specifically deleting the selected point with 'D'."""
        if event.key() == Qt.Key_D and self.selected_point!=None:
            print(f"Deleting Point {self.selected_point} at position {self.points[self.selected_point].position}")
            # Remove the selected point from the view and the list
            self.video_view.removeItem(self.points[self.selected_point])  # Remove point from the view
            self.points.remove(self.points[self.selected_point])  # Remove point from the points list
            self.selected_point = None  # Deselect the point after deleting


# def run_keypoint():
#     # app = QtWidgets.QApplication(sys.argv)
#
#     dialog = DraggablePointDialog()
#     dialog.show()
#
# #     # sys.exit(app.exec_())
