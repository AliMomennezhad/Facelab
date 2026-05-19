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


Brightness ROI
Using “Circle” ROI(s) brighness changes withing the ROI(s) can be measured.  



![image alt](https://github.com/AliMomennezhad/Facelab/blob/main/roi%20blinking1.png)

