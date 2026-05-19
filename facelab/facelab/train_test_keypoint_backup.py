from io import StringIO

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from keypoint_dataset import KeypointDataset
from keypoint_model_training import Keypoint_Class
import matplotlib.pyplot as plt

save_model=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Keypoint_Class().to(device)
# model = Keypoint_Class_two().to(device)
# model=Keypoint_Class(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

all_data = np.load(r"C:\VideoAnalysis\keypointdata\kepoint_data.npy", allow_pickle=True).item()
images = all_data["frame"]
keypoints = all_data["keypoint"]
train_images, test_images, train_keypoints, test_keypoints = train_test_split(images, keypoints, test_size=0.2)
train_loader = DataLoader(KeypointDataset(images=train_images,keypoints=train_keypoints), batch_size=8, shuffle=True)
test_loader = DataLoader(KeypointDataset(images=test_images,keypoints=test_keypoints), batch_size=8, shuffle=False)

# Training loop
def train(model, train_loader, criterion, optimizer, epochs):

    n_factor = 2 ** 4 // (2 ** model.n_upsample) #2**4=16, 2**2=4
    xmesh, ymesh = np.meshgrid(np.arange(256 / n_factor),np.arange(256 / n_factor))
    ymesh = torch.from_numpy(ymesh).to(device)
    xmesh = torch.from_numpy(xmesh).to(device)
    sigma = 3 * 4 / n_factor
    Lx = 64
    LR = 0.001 * np.ones(10, )
    LR[-6:-3] = 0.001 / 10
    LR[-3:] = 0.001 / 25
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
            # print("tell me about images",images.shape)
            # print("tell me about lbl", lbl.shape)
            hm_pred, locx_pred, locy_pred = model(images)
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

            # relative locationsof keypoints
            locx = ymesh - x_true.unsqueeze(-1).unsqueeze(-1) #unsqueeze (-1) add one dimension at the end. (8,4) will be (8,4,1)
            locy = xmesh - y_true.unsqueeze(-1).unsqueeze(-1)
            # print("locy",locy.shape,locy)
            # print("locx", locx.shape,locx)
            # print("x_true", x_true.unsqueeze(-1).unsqueeze(-1).shape)
            # print("y_true", x_true.unsqueeze(-1).unsqueeze(-1).shape)
            # print("ymesh",ymesh)
            # normalize the true heatmaps
            hm_true = torch.exp(-(locx**2 + locy**2) / (2 * sigma**2))
            hm_true = (10* hm_true/ (1e-3 + hm_true.sum(axis=(-2, -1)).unsqueeze(-1).unsqueeze(-1))) #axis=(-1,-2) meansalong the last and last-but-one dimensions
            # mask over which to train the location graphs
            mask = (locx**2 + locy**2) ** 0.5 <= sigma
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
            hm_pred = hm_pred[lbl_nan]
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
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            # this operation clips the gradient and returns its original norm
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
            # keep track of the largest gradient norm on this epoch
            gnorm_max = np.maximum(gnorm_max, gnorm.cpu())

            optimizer.step()

            n_batches += 1

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")



        # for batch in train_loader:
        #     images = batch["image"].to(device)
        #     heatmaps = batch["heatmaps"].to(device)
        #     # print("heatmap",heatmaps.shape)
        #
        #     # Forward pass
        #     outputs = model(images)
        #     print("shape of outputs",[o.shape for o in outputs])
        #     print("shape of outputs[0]", outputs[0].shape)
        #     if isinstance(outputs, (list, tuple)):
        #         outputs = outputs[0]  # Use the last tensor (final predictions)
        #
        #     loss = criterion(outputs, heatmaps)
        #
        #     # Backward pass
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #
        #     running_loss += loss.item()
        #
        # print(f"Epoch {epoch + 1}/{epoch}, Loss: {running_loss / len(train_loader):.4f}")



# Training loop
# def train(model, data_loader, criterion, optimizer, epochs):
#     model.train()
#     progress_output = StringIO()
#     for epoch in tqdm(range(epochs), file=progress_output):
#         print("epoch",epoch)
#         epoch_loss = 0
#         for batch in data_loader:
#             images = batch["image"].to(device)
#             heatmaps = batch["heatmaps"].to(device)
#             # images, keypoint_maps = images.to(device), keypoint_maps.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(images)
#             # print("shape of outputs", [o.shape for o in outputs])
#             #             print("shape of outputs[0]", outputs[0].shape)
#             loss = criterion(outputs, heatmaps)
#             loss.backward()
#             optimizer.step()
#
#             epoch_loss += loss.item()
#
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(data_loader)}")




train(model, train_loader, criterion, optimizer, epochs=500)
# # model.save(r"C:\LocalExpData\blinking2\2025-04-22_1\concatenated\mymodel.h5")

def plot_heatmap(counter,image, heatmaps, true_keypoints=None, pred_keypoints=None, sigma=2):

    image = image.squeeze().cpu().numpy()  # remove dimensions of size 1 from a tensor [1,256,256] will be [256,256]
    heatmaps = heatmaps.cpu().numpy()
    true_keypoints = true_keypoints.cpu().numpy()
    pred_keypoints = pred_keypoints.cpu().numpy()

    num_keypoints = heatmaps.shape[0]
    height, width = image.shape

    print("counter",counter)
    fig, axes = plt.subplots(1, num_keypoints + 1, figsize=(15, 5))
    fig.suptitle(f"Main Title for the Figure{counter}", fontsize=16)
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    # Plot heatmaps
    for i in range(num_keypoints):
        axes[i + 1].imshow(image, cmap='gray', alpha=0.5)
        axes[i + 1].imshow(heatmaps[i], cmap='jet', alpha=0.5)
        axes[i + 1].set_title(f"Heatmap {i+1}")
        axes[i + 1].axis('off')

        # Overlay true keypoints
        if true_keypoints is not None:
            y, x = true_keypoints[i]
            axes[i + 1].scatter(x, y, color='green', s=30, label="True", edgecolor='black')

        # Overlay predicted keypoints
        if pred_keypoints is not None:
            y, x = pred_keypoints[i]
            axes[i + 1].scatter(x, y, color='red', s=30, label="Predicted", edgecolor='black')

def heatmaps_to_keypoints(heatmaps):

    batch_size, num_keypoints, height, width = heatmaps.shape
    keypoints = []

    for b in range(batch_size):
        keypoints_per_image = []
        for k in range(num_keypoints):
            # Flatten heatmap and get the index of the maximum value
            heatmap = heatmaps[b, k]
            max_idx = torch.argmax(heatmap)
            y, x = divmod(max_idx.item(), width)  # Convert to 2D coordinates
            keypoints_per_image.append([x, y])
        keypoints.append(keypoints_per_image)

    return torch.tensor(keypoints, dtype=torch.float32)

def compute_mse(pred_keypoints, true_keypoints):

    total_mse = 0.0
    num_points = 0

    for i in range(pred_keypoints.shape[0]):  # Iterate over batch
        for j, (p, t) in enumerate(zip(pred_keypoints[i], true_keypoints[i])):
            print(f"Batch {i}, Keypoint {j}, Predicted: {p}, True: {t}")  # Debugging
            total_mse += (p[0] - t[0]) ** 2 + (p[1] - t[1]) ** 2
            num_points += 1

    return total_mse / num_points


def compute_pck(pred_keypoints, true_keypoints, threshold, image_size):

    correct_keypoints = 0
    total_keypoints = 0

    # Compute image diagonal for threshold scaling
    diag = np.sqrt(image_size[0] ** 2 + image_size[1] ** 2)
    dist_threshold = threshold * diag

    for pred, true in zip(pred_keypoints, true_keypoints):
        for p, t in zip(pred, true):
            distance = np.sqrt((p[0] - t[0]) ** 2 + (p[1] - t[1]) ** 2)
            if distance <= dist_threshold:
                correct_keypoints += 1
            total_keypoints += 1

    return correct_keypoints / total_keypoints

if save_model:
   torch.save(model.state_dict(), r"C:\VideoAnalysis\keypointdata\mymodel.pt")
else:
    model = Keypoint_Class().to(device)
    # model=torch.load(r"C:\LocalExpData\blinking2\2025-04-22_1\concatenated\mymodel.pt")
    model.load_state_dict(torch.load(r"C:\LocalExpData\blinking2\2025-04-22_1\concatenated\mymodel.pt",map_location=torch.device('cpu')))
    ##important note: don't write: model=model.load_state_dict... it gives you error
print("model is",model)

model.eval()
for batch in test_loader:
    images = batch["image"].to(device)
    print("images",images.shape)
    # true_heatmaps = batch["heatmaps"].to(device)
    true_keypoints = batch["keypoints"]  # True keypoints in coordinate format
    lx = 64
    ly = 64
    batch_size = 8
    num_keypoints = 3
    locx_mesh, locy_mesh = torch.meshgrid(torch.arange(images.shape[0]), torch.arange(num_keypoints), indexing="ij")
    locx_mesh = locx_mesh.to(device)
    locy_mesh = locy_mesh.to(device)

    # Predict
    with torch.no_grad():
        hm_pred, locx_pred, locy_pred = model(images)
        print("h_pred", hm_pred.shape, "lx", lx)

        # if smooth:
        #     hm_pred = gaussian_filter(hm_pred.cpu().numpy(), [0, 1, 1])

        hm_pred = hm_pred.reshape(hm_pred.shape[0], num_keypoints, lx * ly)
        locx_pred = locx_pred.reshape(hm_pred.shape[0], num_keypoints, lx * ly)
        locy_pred = locy_pred.reshape(hm_pred.shape[0], num_keypoints, lx * ly)

        # likelihood, imax = torch.max(hm_pred, -1)
        _, imax = torch.max(hm_pred, -1)
        likelihood = torch.sigmoid(hm_pred)
        likelihood, _ = torch.max(likelihood, -1)
        i_y = imax % lx
        i_x = torch.div(imax, lx, rounding_mode="trunc")
        x_pred = (locx_pred[locx_mesh, locy_mesh, imax] * (-2 * 3)) + i_x
        y_pred = (locy_pred[locx_mesh, locy_mesh, imax] * (-2 * 3)) + i_y

    print("images",images.shape)
    x_pred *= 4
    y_pred *= 4
    row_num = images.shape[0] // 3
    reminder = images.shape[0] % 3
    if row_num != 0 and reminder > 0: row_num = row_num + 1
    col_count=0
    row_count=0
    if row_num==0:
       fig,ax=plt.subplots(max(1,row_num),reminder,figsize=(10,6))
    else:fig,ax=plt.subplots(max(1,row_num),3,figsize=(10,6))
    print("row_num",row_num)

    for i in range(images.shape[0]):
        img=images[i].squeeze().cpu().numpy()
        if row_num !=0:
            # cv2.circle(img,(x_pred[i][0],y_pred[i][0]),radius=5,color=(70,50,60),thickness=-1)
            # cv2.circle(img, (x_pred[i][1], y_pred[i][1]), radius=5, color=(170, 50, 60), thickness=-1)
            # cv2.circle(img, (x_pred[i][2], y_pred[i][2]), radius=5, color=(70, 150, 60), thickness=-1)
            # cv2.circle(img, (x_pred[i][3], y_pred[i][3]), radius=5, color=(70, 50, 160), thickness=-1)
            ax[row_count,col_count].imshow(img,cmap='gray')
            ax[row_count,col_count].scatter(y_pred[i], x_pred[i], color=['orange','blue','yellow'][i], s=30, label="True", edgecolor='black')
        else:
            ax[col_count].imshow(img,cmap='gray')
            ax[col_count].scatter(y_pred[i], x_pred[i], color=['orange','blue','yellow'][i], s=30, label="True", edgecolor='black')
        col_count +=1
        if col_count==3:
            col_count=0
            row_count +=1
    # plt.tight_layout()




    print("yshape", y_pred.shape, "xshape", x_pred.shape)
    print("yshape", y_pred, "xshape", x_pred)
    print("lx", lx, "ly", ly)
    print("true_keypoints",true_keypoints)


plt.show()


# # Evaluation
# model.eval()
# mse_total = 0.0
# pck_total = 0.0
# threshold = 0.05  # 5% of the image diagonal
# image_size = (256, 256)
# counter=0
# with torch.no_grad():
#     for batch in test_loader:
#
#         images = batch["image"].to(device)
#         # true_heatmaps = batch["heatmaps"].to(device)
#         true_keypoints = batch["keypoints"]  # True keypoints in coordinate format
#
#         # Predict heatmaps
#         pred_heatmaps = model(images)
#         if isinstance(pred_heatmaps, tuple):
#             pred_heatmaps = pred_heatmaps[0]  # Adjust index based on model behavior
#         else:
#             pred_heatmaps = pred_heatmaps
#
#         pred_keypoints = heatmaps_to_keypoints(pred_heatmaps)
#
#         plot_heatmap(counter,
#             image=images[0],  # Grayscale image
#             heatmaps=pred_heatmaps[0],  # Heatmaps for the first image
#             true_keypoints=true_keypoints[0],  # True keypoints
#             pred_keypoints=pred_keypoints[0]  # Predicted keypoints
#         )
#         counter =counter+1
#         print("pred_keypoints shape",pred_keypoints.shape)
#         print("pred_keypoints type",type("pred_keypoints"))
#         print("true keypoint",true_keypoints)
#
#         # Compute metrics
#         mse = compute_mse(pred_keypoints, true_keypoints)
#         pck = compute_pck(pred_keypoints, true_keypoints, threshold, image_size)
#
#         mse_total += mse
#         pck_total += pck
#
# # Average metrics
# mse_avg = mse_total / len(test_loader)
# pck_avg = pck_total / len(test_loader)
#
# print(f"Mean Squared Error (MSE): {mse_avg:.4f}")
# print(f"Percentage of Correct Keypoints (PCK): {pck_avg * 100:.2f}%")
# plt.tight_layout()
# plt.show()