import glob
import os.path
from pathlib import Path
import h5py
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cv2 import imread
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.io import ImageCollection
from skimage.registration import phase_cross_correlation
# print(np.arange(-20,40,20))
# res=np.load(r"C:\LocalExpData\blinking2\2025-04-22_1\blinking2_2025-04-22_1_concatenated_extracted_ssim.npy",allow_pickle=True).item()
# print(res)
import pyqtgraph as pg
import torch
from skimage import color
from skimage.feature import hog
from skimage import data, exposure, io
import matplotlib.pyplot as plt
frame_proro=np.load(r'C:\LocalExpData\prototype2\2025-11-07_1\1\prototype2_2025-11-07_1_frame_protfo.npy',allow_pickle=True)
print(frame_proro)


exit()
print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
exit()
img=cv2.imread(r"C:\LocalExpData\pav5\2025-09-04_1\1\masked_14352.png")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img=np.reshape(img,(1,-1)).squeeze(0)
#
#
# print(sorted(img,reverse=True))
x,y=np.where(img>120)
# print(img.index(np.max(img)))
img[x,y]=np.min(img)

plt.imshow(img,cmap='gray')
# print(img)
plt.show()


with h5py.File(r"C:\LocalExpData\dat015\2025-08-05_2\meta_key.h5", 'r') as f:
    for find,fval in f['Facelab'].items():
        print(find,fval[:])



exit()
# expfo=np.load(r"C:\LocalExpData\pav3\2025-09-06_1\concatenated\cam1_pav3_2025-09-09_1_concatenated.npy",allow_pickle=True)
exxxxppp=[
    [r'C:\\LocalExpData\\pav3\\2025-09-09_1\\1\\cam1_2025-09-09_1_1_pav3_eye.avi',
  '31609'],
 [r'C:\\LocalExpData\\pav3\\2025-09-09_1\\1\\cam1_2025-09-09_1_2_pav3_eye.avi',
  '31609'],
 [r'C:\\LocalExpData\\pav3\\2025-09-09_1\\1\\cam1_2025-09-09_1_3_pav3_eye.avi',
  '31607'],
 [r'C:\\LocalExpData\\pav3\\2025-09-09_1\\1\\cam1_2025-09-09_1_4_pav3_eye.avi',
  '31608'],
 [r'C:\\LocalExpData\\pav3\\2025-09-09_1\\1\\cam1_2025-09-09_1_5_pav3_eye.avi',
  '31608'],
 [r'C:\\LocalExpData\\pav3\\2025-09-09_1\\1\\cam1_2025-09-09_1_6_pav3_eye.avi',
  '31608'],
 [r'C:\\LocalExpData\\pav3\\2025-09-09_1\\1\\cam1_2025-09-09_1_7_pav3_eye.avi',
  '31612'],
    [r'C:\\LocalExpData\\pav3\\2025-09-09_1\\1\\cam1_2025-09-09_1_8_pav3_eye.avi',
  '31606'],
    [r'C:\\LocalExpData\\pav3\\2025-09-09_1\\1\\cam1_2025-09-09_1_9_pav3_eye.avi',
  '31615'],
    [r'C:\\LocalExpData\\pav3\\2025-09-09_1\\1\\cam1_2025-09-09_1_10_pav3_eye.avi',
  '31606'],
    [r'C:\\LocalExpData\\pav3\\2025-09-09_1\\1\\cam1_2025-09-09_1_11_pav3_eye.avi',
  '31613'],
    [r'C:\\LocalExpData\\pav3\\2025-09-09_1\\1\\cam1_2025-09-09_1_12_pav3_eye.avi',
  '31614'],
    [r'C:\\LocalExpData\\pav3\\2025-09-09_1\\1\\cam1_2025-09-09_1_13_pav3_eye.avi',
  '31613']]
np.save(r"C:\LocalExpData\pav3\2025-09-06_1\concatenated\cam1_pav3_2025-09-09_1_concatenated.npy",exxxxppp)
print(np.load(r"C:\LocalExpData\pav3\2025-09-06_1\concatenated\cam1_pav3_2025-09-09_1_concatenated.npy",allow_pickle=True))
a=[1,2,3,4,5]
b=[7,5,0,9,3]
callc=(np.array(a)-np.array(b))**2
print(callc)
exit()

plt.plot([1,2,3,-4,6],'b',label="blue")
plt.plot([-1,2,-3,4,9],'r',label="red")
plt.legend()
plt.show()

img_name=io.ImageCollection(r"C:\LocalExpData\dat015\2025-08-15_1\Frame Extraction\cam1_airpuff\*.jpg").files
event_onsets=np.unique([int(img.split('(')[1].split(')')[0]) for img in img_name])
event_lengths=[int(img.split('(')[1].split(')')[0]) for img in img_name]
print(event_lengths)
event_frames=[]
for ev_ons in event_onsets:
    event_frames.append([[],[]])
    for img_nm_ind,img_nm_val in enumerate(img_name):
        if ev_ons==event_lengths[img_nm_ind]:
            if "before" in img_nm_val:
               event_frames[-1][0].extend([img_nm_val])
            else:event_frames[-1][1].extend([img_nm_val])

print("before",event_frames[0][0])
print("after",event_frames[0][1])
exit()
# Loading an example image
from sklearn.metrics.pairwise import cosine_similarity
experiment_information=np.load(r"C:\LocalExpData\dat015\2025-08-15_1\dat015_2025-08-15_1_frame_protfo.npy",allow_pickle=True)
frame_exclude=experiment_information[0][3]
print(frame_exclude)
exclusion_arr=[]
for f_ex0 in frame_exclude:
    for f_ex in f_ex0[1]:
        exclusion_arr.extend(np.arange(f_ex[0]-3,f_ex[1]+2,1))


print(exclusion_arr)


exit()

image=imread(r"C:\LocalExpData\dat015\2025-08-15_1\Frame Extraction\cam1_airpuff\airpuff (5475)_after_5475.jpg")
image1=imread(r"C:\LocalExpData\dat015\2025-08-15_1\Frame Extraction\cam1_airpuff\airpuff (5475)_after_5476.jpg")
image2=imread(r"C:\LocalExpData\dat015\2025-08-15_1\Frame Extraction\cam1_airpuff\airpuff (5475)_after_5542.jpg")

imagefiles=io.ImageCollection(r"C:\LocalExpData\dat015\2025-08-15_1\Frame Extraction\cam1_airpuff\*.jpg")[:10]
imagefiles_name=io.ImageCollection(r"C:\LocalExpData\dat015\2025-08-15_1\Frame Extraction\cam1_airpuff\*.jpg").files[:10]
image_name=zip(imagefiles,imagefiles_name)


ref_features, ref_hog_image = hog(imagefiles[0], orientations=9, pixels_per_cell=(8, 8),cells_per_block=(30, 30), visualize=True)
cosine_sim=[]
for img_file,img_name in image_name:

    # image = cv2.cvtColor(imread(img_file), cv2.COLOR_RGB2GRAY)
    features, hog_image = hog(img_file, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(30, 30), visualize=True)
    print("name is", img_name, 'cosine', cosine_similarity(ref_features.reshape(1, -1), features.reshape(1, -1)),'euc',np.linalg.norm(ref_features-features))
    cosine_sim.extend(cosine_similarity(ref_features.reshape(1, -1), features.reshape(1, -1)))

print(cosine_sim)
plt.figure()
plt.plot(cosine_sim)
plt.xticks(np.arange(0,101,1),np.arange(5475,5475+101,1),fontsize=8)
plt.show()



image2=cv2.cvtColor(image2,cv2.COLOR_RGB2GRAY)
image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
image1=cv2.cvtColor(image1,cv2.COLOR_RGB2GRAY)
features, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(30, 30), visualize=True)
features2, hog_image2 = hog(image2, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(30,30), visualize=True)
features1, hog_image1 = hog(image1, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(30,30), visualize=True)
print(cosine_similarity(features.reshape(1,-1),features2.reshape(1,-1)))
print(cosine_similarity(features.reshape(1,-1),features1.reshape(1,-1)))
plt.figure()
plt.imshow(image2,cmap='gray')
plt.figure("normal")
plt.imshow(hog_image,cmap='gray')
plt.figure("paw")
plt.imshow(hog_image2,cmap='gray')
plt.show()

from sklearn.ensemble import RandomForestClassifier

teee=[1,2,3,4,5,6,7,8,9,10,11,12]
print(teee[::3])
exit()

helloo=[[1,2,3,"sdf"],[5,4,5,"hssh"]]
print(np.delete(helloo,[0,1],axis=0))
exit()
kepoint=np.load(r"C:\LocalExpData\DBH0015\2025-07-28_1\kepoint_data.npy",allow_pickle=True).item()
print(np.shape(kepoint['frame'][40]))
print(kepoint['keypoint'][40])
plt.imshow(kepoint['frame'][40],cmap='gray')
plt.scatter(kepoint['keypoint'][40][1],kepoint['keypoint'][40][0],color='g',s=30)
plt.show()


image = cv2.imread(r'C:\LocalExpData\DBH0015\2025-07-03_2\concatenated/masked_172562.jpg')
image_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) # Converting image to grayscale
event_fr= [2510, 5475, 8441, 11406, 14372, 17338, 20304, 23270]
event_frames=[]
for evntt in event_fr:
    event_frames.extend(np.arange(evntt-500,evntt+200,1))
print(event_frames)
exit()

model=RandomForestClassifier
model.fit
importance=model.feature_importances_
# Extract HOG features
features, hog_image = hog(image_gray, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(1, 1), visualize=True)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image_gray, cmap='gray')
plt.title('original image')

plt.subplot(1, 2, 2)
plt.imshow(hog_image, cmap='gray')
plt.title('hog')
plt.show()

temmp=np.load(r"C:\LocalExpData\blinking2\2025-04-22_1\concatenated\meta_key.npy",allow_pickle=True)
for items in temmp:
    for item in items:
        print(item[0])

exit()


import h5py
with h5py.File(r"C:\VideoAnalysis\blinking26_2025-05-16_1_concatenatedDLC_resnet50_blinking26_deeplabcutJun11shuffle1_100000.h5",'r') as file:
    deeplab_x=  [i[1][0] for i in file['df_with_missing']['table']]
    deeplab_y = [i[1][1] for i in file['df_with_missing']['table']]
    plt.figure()
    plt.plot(deeplab_y,"r")
    # plt.plot(deeplab_x, "b")
    plt.show()


exit()
import cv2
import numpy as np
chk=[[1,2,3]]
for i in range(0,3):
    print([5,9,7,8][i])
multi_arr=np.array([[[8,9,6,4],[8,9,6,4],[8,9,6,4]],
           [[8,9,6,4],[8,9,6,4],[8,9,6,4]]])
print("shape of multi",multi_arr.shape)
fig=plt.figure()
plt.legend
x,y=np.where(np.array([5,2,6,9,4,7])>2)
print(x,y)
for i in range(1,len(chk)):
    print(i)
print(np.arange(600,607)[0:2])
lii=[[1],[-9,-8,-2,-4]]
for li1 in lii:
    if len(li1)==1:
        li1.extend(np.arange(li1[0]+1,li1[0]+5))
    li1.extend([10])
    print(li1)


print([np.arange(600,611)])
np.linspace
plt.show()
cats=['one','two','three']
val=[0,10,15]
del val[val.index(10)]
print(val)
exit()
for cat in range(len(cats)):
    plt.bar(cats[cat],[0,10,15][cat],facecolor='none',edgecolor=['b','r','y'][cat])
    plt.text(cats[cat],[0,10,15][cat]+1,val[cat],va='bottom',ha='center')
plt.yticks([0,10,20,30,40,50])
plt.tight_layout()
plt.show()

arr=[0,1,2,3,6,8,0,10,15,18,19,-10,14,19,30,155,0,12,16,50,90,2]

baseline=pd.DataFrame(data=arr).rolling(window=5).median()
print("baseline",baseline)

import matplotlib.pyplot as plt
plt.text(0.5, 0.5, 'hello'+'\n'+'konichiwa', fontsize=12, ha='center', va='center', wrap=True)
plt.axis('off')
plt.show()
def show_text_plot(text):
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, text, fontsize=12, ha='center', va='center', wrap=True)
    ax.axis('off')
    plt.show()

# Example usage
show_text_plot("Here is your text shown in a matplotlib-style window.bnfxghgfhfgdhftgsdhdfgthhhhhhhhhhhhhhhhhhhhhhhhhhhhh")


plt.figure()
plt.plot(arr)
plt.axvline(np.median(arr))
plt.show()

new_arr=[arr[i]-arr[i-2] for i in range(2,len(arr)) ]
print([0,0]+new_arr)
find_peaks()

exit()

aliis=[1,2,4,5,6,7,8,9,11,15,18,51,64]
aliis1=[10,20,40,1,6,7,8,2,11,15,18,51,64]
for i in aliis:
    print(i)
    for i1 in aliis1:
        print(i,i1)
        if i==i1:
            break
print("fini")
exit()

image_file_path=os.path.join(r'C:\LocalExpData\blinking 26\2025-05-20_1\1','*.jpg')
image_collection=ImageCollection(image_file_path).files
img_arr=[]
shift_arr=[]
for img in image_collection:
    img_arr.append(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
print(len(img_arr))
for i in range(len(img_arr)):
    img1=np.float32(img_arr[min(0,i-1)])
    img2 = np.float32(img_arr[i])
    hanning = cv2.createHanningWindow(img1.shape[::-1], cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(img1, img2, hanning)
    shift_arr.append(shift)
    print("Estimated shift (x, y):", shift)
    print("Correlation strength (confidence):", response)
shift_x=[x[0] for x in shift_arr]
shift_y=[x[1] for x in shift_arr]
print(shift_x)
print(shift_y)
plt.plot(shift_x,'-*r')
plt.plot(shift_y,'-^b')
plt.show()

print(image_collection)
# Load two grayscale images (frames)



print(np.random.randint(0,255,3))
# a = [1, 2, 3, 4, 5]
# b = [2, 3, 4, 5, 6]
#
from tqdm import tqdm
from io import StringIO

nsegs = 1000
s = StringIO()

for i in tqdm(range(nsegs), desc="Processing", file=s):
    _=i**2



# Access the raw tqdm output
captured = s.getvalue()
print(s)
print("Captured tqdm output:")
print(captured)

# t_stat, p_val = stats.ttest_ind(a,b)
# print(f"T-statistic: {t_stat}, P-value: {p_val}")
# rng = np.random.default_rng()
# rvs = stats.uniform.rvs(size=10, random_state=rng)
# stats.ttest_1samp()
# stats.ttest_rel()
# print(rvs)
#
# print([i for i,val in enumerate(res["score"]) if val<=0.2])
#[5476, 5477, 5478, 5479, 5480, 5481, 5482, 5483, 5484, 5485, 5486, 5487, 5488, 5489, 5490, 5491, 5492, 5493, 5494, 5518, 5519, 5520, 5521, 5522, 5523, 5524, 5525, 5526, 6362, 6363, 6364, 6365, 6366, 6367, 6368, 6369, 6370, 6371, 6372, 6373, 6374, 6375, 6376, 7156, 7157, 7158, 7159, 7160, 7161, 7162, 7163, 7164, 7165, 7166, 7167, 7168, 14356, 14357, 14358, 14359, 14360, 14361, 14362, 14363, 14364, 14365, 14366, 14367, 14368, 14369, 14370, 14371, 14372, 23264, 23265, 23266, 23267, 23268, 23269, 23270, 23271, 23272, 23273, 23274, 23275, 23276, 23277, 23278, 23279, 23280, 40047, 40048, 40049, 40050, 40051, 40052, 40053, 40054, 40055, 40056, 40057, 40058, 40059, 40060, 40061, 40062, 40063, 40064, 40065, 40066, 40068, 40089, 40090, 40091, 40092, 40093, 40094, 40095, 40096, 40097, 40098, 40099, 40100, 40101, 40102, 40103, 40104, 40105, 40106, 40107, 40109, 40110, 40179, 40180, 48943, 48944, 48945, 48946, 48947, 48948, 48949, 48950, 48951, 48952, 48953, 48954, 48955, 48956, 48957, 48958, 48959, 48960, 48961, 54872, 54873, 54874, 54875, 54876, 54877, 54878, 54879, 54880, 54881, 54882, 54883, 54884, 54885, 54886, 54887, 54888, 54889, 54890, 54891, 54892, 54893, 54894, 54895, 54896, 54897, 54898, 54899, 54900, 54901, 54902, 54920, 65726, 65727, 65728, 65729, 65730, 65731, 65732, 65733, 65734, 65735, 65736, 65737, 65738, 65739, 65740, 65741, 65742, 65743, 65744, 68693, 68694, 68695, 68696, 68697, 68698, 68699, 68700, 68701, 68702, 68703, 68704, 68705, 68706, 68707, 68711, 68712, 68713, 68714, 68715, 68716, 68717, 68718, 68719, 68720, 68721, 68722, 68723, 68724, 68725, 68726, 68727, 69263, 70135, 70136, 70137, 86485, 86486, 86487, 86488, 86489, 86490, 86491, 86492, 86493, 86494, 86495, 86496, 86497, 86498, 86500, 86582, 86583, 86584, 86585, 86586, 86587, 86588, 86589, 86590, 86591, 86592, 86593, 86594, 86595, 86596, 90825, 90826, 90827, 90828, 90829, 90830, 90831, 90832, 90833, 90834, 90835, 90836, 90837, 90838, 90839, 90840, 90841, 90842, 100305, 100306, 100307, 100308, 100309, 100310, 100311, 100312, 100313, 100314, 100315, 100316, 100317, 100318, 100319, 100320, 100321, 100322, 100323, 100324, 100325, 100326, 106237, 106238, 106239, 106240, 106241, 106242, 106243, 106244, 106245, 106246, 106247, 106248, 106249, 106250, 106251, 106254, 106255, 106256, 106257, 106258, 106259, 106260, 106261, 106262, 106263, 106264, 106265, 106266, 106267, 106268, 106269, 106270, 106271, 106273, 106274, 106275, 106276, 106277, 106278, 106279, 106280, 106281, 106282, 106283, 112165, 112166, 112167, 112168, 112169, 112170, 112171, 112172, 112173, 112174, 112175, 112176, 112177, 112178, 112179, 112180, 112181, 112182, 112183, 112184, 112185, 112186, 112187, 112188, 143780, 143781, 143782, 143783, 143784, 143785, 143786, 143787, 143788, 143789, 143790, 143791, 143792, 143793, 143794, 143795, 143796, 143797, 143798, 143799, 143800, 143825, 143826, 143827, 143828, 143829, 143830, 143831, 143832, 143833, 143834, 143835, 143836, 143837, 143838, 143839, 143840, 143841, 143842, 143843, 143844, 143900, 143901, 149067, 149068, 149069, 149070, 149071, 149072, 149073, 149074, 149075, 149076, 149077, 149078, 149079, 149080, 149081, 149082, 149083, 149084, 149085, 149086, 149712, 149713, 149714, 149715, 149716, 149717, 149718, 149719, 149720, 149721, 149722, 149723, 149724, 149725, 149726, 149728, 155646, 155647, 155648, 155649, 155650, 155651, 155652, 155653, 155654, 155655, 155656, 155657, 155658, 155659, 155660, 155661, 155662, 155663, 155664, 155665, 155666, 155667, 155668, 155669, 155670, 155671, 155672, 155673, 155674, 155675, 157692, 157693, 157694, 157695, 157696, 157697, 157698, 157699, 157700, 157701, 157702, 157703, 157704, 157705, 157706, 157707, 157708, 157709, 169463, 169464, 169465, 169466, 169467, 169468, 169469, 169470, 169471, 169472, 169473, 169474, 169475, 169476, 169477, 169478, 169479, 169480, 169481, 169482, 169483, 169484, 169485, 169486, 169487, 169510, 169511, 169512, 169513, 169514, 169515, 169516, 169517, 169518, 169519, 169520, 169521, 169522, 169523, 169524, 169525, 169526, 169527, 169528, 172425, 172426, 172427, 172428, 172429, 172430, 172431, 172432, 172433, 172434, 172435, 172436, 172437, 172438, 172439, 172440, 172441, 172442, 172443, 172444, 172464, 172465, 172466, 172467, 172468, 172469, 172470, 172471, 172472, 172473, 172474, 172475, 172476, 172477, 172478, 172479, 175403, 175404, 175405, 175406, 175407, 175408, 175409, 175410, 175411, 175412, 175413, 175414, 175415, 175416, 175417, 175418, 175419, 175420, 175421, 175422, 175423, 175425, 175426, 192178, 192179, 192180, 192181, 192182, 192183, 192184, 192185, 192186, 192187, 192188, 192189, 192190, 192191, 192192, 192193, 192194, 192195, 192196, 192197, 192198, 192199, 192200, 192205, 192206, 192207, 192208, 192209, 192210, 192211, 192212, 192213, 192214, 192215, 192216, 192217, 192218, 192219, 192220, 192221, 192278, 192279, 192348, 201077, 201078, 201079, 201080, 201081, 201082, 201083, 201084, 201085, 201086, 201087, 201088, 201089, 201090, 201114, 201115, 201116, 201117, 201118, 201119, 201120, 201121, 201122, 201123, 201124, 201125, 201126, 201127, 201128, 201129, 201130, 201131, 201132, 201133, 218880, 218881, 218882, 218883, 218884, 218885, 218886, 218887, 218888, 218889, 218890, 218891, 218892, 218893, 218894, 218895, 218896, 218897, 218898, 218899, 218900, 218927, 218928, 218929, 218930, 218931, 218932, 218933, 218934, 218935, 218936, 218937, 218938, 218939, 223174, 223175, 223176, 223177, 223178, 223179, 223180, 223181, 223182, 223183, 223184, 223185, 223186, 223187, 223210, 223211, 223212, 223213, 223214, 223215, 223216, 223217, 223218, 223219, 223220, 223221, 223801, 223802, 223803, 223804, 223805, 223806, 223807, 223808, 223809, 223810, 223811, 223812, 223813, 223814, 223815, 223816, 223817, 223819, 223820, 223821, 223822, 223823, 223824, 223825, 223826, 223827, 223828, 226694, 226695, 226696, 226697, 226698, 226699, 226700, 226701, 226702, 226703, 226704, 226705, 226706, 226707, 226708, 226709, 226710, 226711, 226712, 226713, 226714, 226715, 226716, 226717, 226718, 226719, 238622, 238623, 238624, 238625, 238626, 238627, 238628, 238629, 238630, 238631, 238632, 238633, 238634, 238635, 238636, 238637, 238639, 238640, 238641, 238642, 238643, 238644, 238645, 238646, 238647, 238648, 238649, 238650, 238651, 238652, 238653, 238654, 238655, 238656, 238657, 238658, 238659, 238662, 238663, 238664, 238665, 238666, 238667, 238668, 238669, 238670, 238671, 238672, 238673, 238674, 238675, 238676, 244556, 244557, 244558, 244559, 244560, 244561, 244562, 244563, 244564, 244565, 244566, 244567, 244701, 244702, 244703, 244704, 244705, 244706, 244707, 244708, 244709, 244710, 244711, 244712, 244713, 244714, 244715, 244716, 244717, 244718, 244719, 244720, 244721, 244722, 244723, 255411, 255412, 255413, 255414, 255415, 255416, 255417, 255418, 255419, 255420, 255421, 255422, 255423, 255424, 255425, 255426, 255427, 255428, 255429, 255430, 255431, 255432, 255433, 255434, 255435, 255436, 258382, 258384, 258385, 258386, 258387, 258388, 258389, 258390, 258391, 258392, 258393, 258394, 258395, 258396, 258397, 258398, 258399, 258400, 270243, 270244, 270245, 270246, 270247, 270248, 270249, 270250, 270251, 270252, 270253, 270254, 270255, 270256, 270257, 270258, 270259, 270260, 270261, 270262, 270263, 270264, 270266, 270267, 270268, 277876, 277877, 277878, 277879, 277880, 278089, 278090, 278091, 278092, 278093, 278094, 278095, 278096, 278097, 278098, 278099, 278100, 278101, 278102, 278103, 278104, 278105, 278106, 278107, 278108, 280699, 280700, 280701, 280702, 280703, 280704, 280705, 280706, 280707, 280708, 280709, 280710, 280711, 280712, 280713, 280714, 280715, 280716, 280717, 280718, 280719, 280720, 280721, 280722, 286976, 286977, 286978, 286979, 286980, 286981, 286982, 286983, 286984, 286985, 286986, 286987, 286988, 286989, 286990, 287033, 287034, 287035, 287036, 287037, 287038, 287039, 287040, 287041, 287042, 287043, 287044, 287045, 287046, 310750, 310751, 310752, 310753, 310754, 310755, 310756, 310757, 310758, 310759, 310760, 310761, 310762, 310763, 310764, 310765, 310766, 310767, 310768, 310769, 310770, 310771, 310772, 310773, 310776, 310777, 310778, 310779, 310780, 310781, 310782, 310783, 310784, 310785, 310786, 310787, 310788, 310789, 310790, 310791, 310792, 310795, 310796, 310797, 310798, 310799, 310800, 310801, 310802, 310803, 310804, 310805, 310806, 313715, 313716, 313717, 313718, 313719, 313720, 313721, 313722, 313723, 313724, 313725, 313726, 313727, 313728, 313729, 313730, 313731, 313732, 313733, 313734, 313735, 318647, 318648, 318649, 318650, 318651, 318652, 318653, 318654, 318655, 318656, 318657, 318658, 318659, 318660, 318661, 318662, 320246, 320247, 320248, 320249, 320250, 320251, 320252, 320253, 320254, 320255, 320256, 320257, 320258, 320259, 320260, 321607, 321608, 321609, 321610, 321611, 321612, 321613, 321614, 321615, 321616, 321617, 321618, 321619, 321620, 321621, 321622, 321623, 321624, 321625, 321626, 321628, 321629, 321630, 321631, 321632, 321633, 321634, 321635, 321636, 321637, 321638, 321639, 321640, 321641, 321642, 321643, 342362, 342363, 342364, 342365, 342366, 342367, 342368, 342369, 342370, 342371, 342372, 342373, 342374, 342375, 342376, 342377, 342382, 342383, 342384, 342385, 342386, 342387, 342388, 342389, 342390, 342391, 342392, 342393, 342394, 342395, 342396, 342397, 342475, 347427, 347428, 347429, 347430]

# vid="C:/LocalExpData/blinking2/2025-04-22_1/blinking2_2025-04-22_1_concatenated.avi"
# path = Path(vid)
# parent_file = path.parent
# stem_file = path.stem
# print(parent_file)
# print(stem_file)
# npy_files=glob.glob(os.path.join(parent_file,"*.npy"))
# if r"C:\LocalExpData\blinking2\2025-04-22_1\blinking2_2025-04-22_1_concatenated_extracted_ssim.npy" in npy_files:
#     print("hi")

# img1=imread(r"C:\LocalExpData\blinking 26\2025-05-20_1\1\masked_0.jpg")
# img2=cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
# print(np.shape(img1))
# plt.figure()
# plt.imshow(img1)
# plt.figure()
# print(np.shape(img2))
# plt.imshow(img2[209:261,281:351],cmap='gray')
# plt.show()
#
# calculated_shift, error, diffphase = phase_cross_correlation(image1, image2)
# print(calculated_shift, error, diffphase )
arr=[0,2,10,-1]
print(arr)
for i in range(5,68):
    arr[i] = 0


print(arr)

prev_gray = None
diff_svds = []
cap = cv2.VideoCapture('C:/LocalExpData/blinking2/2025-04-22_1/blinking2_2025-04-22_1_concatenated.avi')
i=0
while i<5500:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        U, S, VT = np.linalg.svd(diff, full_matrices=False)
        diff_svds.append((U, S, VT))
    if i==0:
       prev_gray = gray
    i +=1


singular_values = [S[0] for _, S, _ in diff_svds]  # Top singular value per frame difference
plt.plot(singular_values)
plt.title('Top Singular Value Over Time')
plt.xlabel('Frame')
plt.ylabel('Singular Value')
plt.show()












