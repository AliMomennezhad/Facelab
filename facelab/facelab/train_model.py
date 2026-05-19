import os.path
from io import StringIO

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

from facelab.keypoint_dataset import KeypointDataset
from facelab.keypoint_model_training import Keypoint_Class
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def initiate_training(keypoint_data_path,learning_rate,destination_path=None,model_path=None,parent=None,epoch=None,log_signal=None):
    model = Keypoint_Class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if model_path is not None:
         print("This model was loaded: ",model_path," and device is: ",device)
         model.load_state_dict(torch.load(model_path,map_location=device)["model_state_dict"])
         optimizer.load_state_dict(torch.load(model_path)["optimizer_state_dict"])
         optimizer.param_groups[0]['lr']=learning_rate
         # for param_group in optimizer.param_groups:
         #     param_group['lr']=0.0001
    print('Learning rate is:',optimizer.param_groups[0]['lr'])
    criterion = nn.MSELoss()

    print("Data for training path: ",keypoint_data_path)
    all_data = np.load(os.path.join(keypoint_data_path,'kepoint_data.npy'), allow_pickle=True).item()

    images = all_data["frame"]
    keypoints = all_data["keypoint"]
    train_loader = DataLoader(KeypointDataset(images=images,keypoints=keypoints), batch_size=8, shuffle=True)
    train(model, train_loader, criterion, optimizer, epochs=epoch,logsignal=log_signal)
    #
    # torch.save({"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict()}, destination_path)
    parent.model=model
    parent.optimizer=optimizer
    print("Training Finished!")





# Training loop
def train(model, train_loader, criterion, optimizer, epochs,logsignal):

    n_factor = 2 ** 4 // (2 ** model.n_upsample) #2**4=16, 2**2=4
    xmesh, ymesh = np.meshgrid(np.arange(256 / n_factor),np.arange(256 / n_factor))
    ymesh = torch.from_numpy(ymesh).to(device)
    xmesh = torch.from_numpy(xmesh).to(device)
    sigma = 3 * 4 / n_factor
    Lx = 64
    # LR = 0.001 * np.ones(10, )
    # LR[-6:-3] = 0.001 / 10
    # LR[-3:] = 0.001 / 25
    # model.train()
    progress_output = StringIO()
    for epoch in tqdm(range(epochs),file=progress_output):

        # model.train()
        running_loss = 0.0
        train_mean = 0
        n_batches = 0
        gnorm_max = 0

        for train_batch in train_loader:
            model.train()
            images = train_batch["image"].to(device, dtype=torch.float32)
            lbl = train_batch["keypoints"].to(device, dtype=torch.float32)
            # print("tell me about images", images.shape)
            # print("tell me about lbl", lbl.shape)
            # plt.figure()
            # plt.imshow(images[0][0],cmap='gray')
            # plt.scatter(lbl[0][0][0],lbl[0][0][1], color='green', s=30, label="True", edgecolor='black')
            # plt.scatter(lbl[0][1][0], lbl[0][1][1], color='orange', s=30, label="True", edgecolor='black')
            # plt.scatter(lbl[0][2][0], lbl[0][2][1], color='yellow', s=30, label="True", edgecolor='black')
            # plt.show()

            # print("tell me about lbl", lbl.shape)
            hm_pred, locx_pred, locy_pred = model(images)
            # print("hm_pred",hm_pred,"locx_pred", locx_pred[0], "locy_pred", locy_pred[0])
            ######################################################################

            # do a lot of preparations for the true heatmaps and the location graphs
            # print("what is label here",lbl)
            lbl_mask = torch.isnan(lbl).sum(axis=-1) #the last axis will be removed (lets say: 8,4,2). It will become (8,2)
            # print("what is lbl mask here",lbl_mask)
            lbl[lbl_mask > 0] = 0
            lbl_nan = lbl_mask == 0
            lbl_nan = lbl_nan.to(device=device)
            lbl_batch = lbl
            # print("label batch",lbl_batch)
            # print("label batch[:, :, 0]", lbl_batch[:, :, 0])

            # divide by the downsampling factor (typically 4)
            y_true = (lbl_batch[:, :, 0]) / 4 #x coordinate will be divided by 4. For example: 160/4=40
            x_true = (lbl_batch[:, :, 1]) / 4   #y coordinate will be divided by 4. For example: 200/4=50
            # relative locations of keypoints
            locx = ymesh - x_true.unsqueeze(-1).unsqueeze(-1) #unsqueeze (-1) add one dimension at the end. (8,4) will be (8,4,1)
            locy = xmesh - y_true.unsqueeze(-1).unsqueeze(-1)
            # print("locy",locy.shape,locy)
            # print("locx", locx.shape,locx)
            # print("x_true", x_true)
            # print("y_true", y_true)
            # print("ymesh",ymesh)
            # normalize the true heatmaps
            hm_true = torch.exp(-(locx**2 + locy**2) / (2 * sigma**2))
            hm_true = (10* hm_true/ (1e-3 + hm_true.sum(axis=(-2, -1)).unsqueeze(-1).unsqueeze(-1))) #axis=(-1,-2) means along the last and last-but-one dimensions
            # mask over which to train the location graphs
            mask = (locx**2 + locy**2) ** 0.5 <= sigma #finding the pixels around the desired coordiante not the exact location

            # normalize the location graphs for prediction
            locx = locx / (2 * sigma)
            locy = locy / (2 * sigma)
            # mask out nan's
            hm_true = hm_true[lbl_nan]

            y_true = y_true[lbl_nan]
            x_true = x_true[lbl_nan]
            locx = locx[lbl_nan]
            # print("lbl_nan",lbl_nan)
            # print("locy[lbl_nan]",locy[lbl_nan])
            locy = locy[lbl_nan]
            mask = mask[lbl_nan]

            # subsample the non-nan heatmaps and location graphs

            hm_pred = hm_pred[lbl_nan] #now the shape of hm_pred changes from 8,2,64,64 to 16,64,64

            locx_pred = locx_pred[lbl_nan]
            locy_pred = locy_pred[lbl_nan]

            # heatmap loss
            loss = ((hm_true - hm_pred).abs()).sum(axis=(-2, -1))


            # loss from the location graphs, masked with mask
            # I use a weighting of 0.5. Much smaller or much bigger worked almost as well (0.05 and 5)
            loss += 0.5 * (mask * ((locx - locx_pred) ** 2 + (locy - locy_pred) ** 2) ** 0.5).sum(axis=(-2, -1))

            with torch.no_grad():
                # this part computes the position error on the training set

                hm_pred = hm_pred.reshape(hm_pred.shape[0], Lx * Lx)
                locx_pred = locx_pred.reshape(locx_pred.shape[0], Lx * Lx)
                locy_pred = locy_pred.reshape(locy_pred.shape[0], Lx * Lx)

                nn = hm_pred.shape[0]
                imax = torch.argmax(hm_pred, 1)

                x_pred = (ymesh.flatten()[imax] - (2 * sigma) * locx_pred[np.arange(nn), imax])
                y_pred = (xmesh.flatten()[imax] - (2 * sigma) * locy_pred[np.arange(nn), imax])

                y_err = (y_true - y_pred).abs()
                x_err = (x_true - x_pred).abs()

                train_mean += ((y_err + x_err) / 2).mean().item()
            loss = loss.mean()
            running_loss += loss.item() #for each batch



            # this operation clips the gradient and returns its original norm
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
            # keep track of the largest gradient norm on this epoch
            gnorm_max = np.maximum(gnorm_max, gnorm.cpu())

            #this part is for clearing the previous gradients and avoiding accumulation of them
            optimizer.zero_grad() #first clears the gradients
            loss.backward() #calculate the gradients
            optimizer.step() #updating the weights using the gradients


            n_batches += 1

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")
        logsignal.emit(int(100*(epoch+1)/epochs))






