# Facelab
Facelab GUI is for analyzing facial video recordings. 

1. Clone or download the repository to your PC.

2. Create a virtual environment using Anaconda (Python 3.9 or above is required):
   conda create --name facelab python=3.9

3. Install the package in editable mode:
   pip install -e .

4. Run the application:
   python -m facelab


If your PC has GPU, you can uninstall torch and use pytorch for faster processing.

------------------------------------------------------------------------------------

Main Features:

•	Brightness ROI

•	Exclusion ROI

•	Inclusion ROI

•	ROI Saving

•	ROI Loading 

•	Keypoint Training

•	Keypoint fine tuning

•	Frame Extraction

•	Event Extraction


Brightness ROI:
Using “Circle” ROI(s) brighness changes withing the ROI(s) can be measured.  
![image alt](https://github.com/AliMomennezhad/Facelab/blob/main/blink%20traces1.png)


Exclusion ROI:
Using “Circle” or “Rectangular” ROIs desired segments within the ROIs can be excluded.
![image alt](https://github.com/AliMomennezhad/Facelab/blob/main/roi%20blinking1.png)

Inclusion ROI:
Using “Mask out” ROI desired segments outside the ROI can be excluded.
![image alt](https://github.com/AliMomennezhad/Facelab/blob/main/exclusion-inclusion%20roi.png)

ROI Saving:
ROIs can be saved.
![image alt](https://github.com/AliMomennezhad/Facelab/blob/main/save%20roi.png)

ROI Loading:

ROIs can be loaded.

Keypoint Training:

It has three keypoints. The user can decide to assign each keypoint to a different feature from scratch and build a training model.
![image alt](https://github.com/AliMomennezhad/Facelab/blob/main/keypoint%20training1.png)

Keypoint fine tuning:

After training the model using the keypoints. The model can be fine tuned and its accuracy improved.
![image alt](https://github.com/AliMomennezhad/Facelab/blob/main/fine_tune%20keypoint1.png)

Frame Extraction:

Frames can be extracted while modifying them.
![image alt](https://github.com/AliMomennezhad/Facelab/blob/main/frame%20extratcion1.png)

Event Extraction:

If your experiment has a protocol or structure, the corresponding frames of the events can be modified and extracted for predefined durations.
![image alt](https://github.com/AliMomennezhad/Facelab/blob/main/frame%20extratcion2.png)




