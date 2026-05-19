from facelab.keypoint_model_training import Keypoint_Class
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset



class Keypoint_predict_Dataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):

        image = self.images[idx]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize
        return {"image": image}





def predict_net(input_data,model,device,log=None):
    x_pred_final=[]
    y_pred_final=[]
    likelihood_final=[]

    test_loader = DataLoader(Keypoint_predict_Dataset(images=input_data), batch_size=8, shuffle=False)
    model.eval() #it puts the model in evaluation and test mode
    conter=0
    for batch in test_loader:
        if log is not None:
            conter +=1
            log.emit(str(conter/len(test_loader)))
        images = batch["image"].to(device)
        # true_keypoints = batch["keypoints"].to(device)  # True keypoints in coordinate format
        lx =ly= 64
        batch_size = 8
        num_keypoints = 3
        locx_mesh, locy_mesh = torch.meshgrid(torch.arange(images.shape[0]), torch.arange(num_keypoints), indexing="ij")
        locx_mesh = locx_mesh.to(device)
        locy_mesh = locy_mesh.to(device)

        # Predict
        with torch.no_grad():
            hm_pred, locx_pred, locy_pred = model(images)
            hm_pred = hm_pred.reshape(hm_pred.shape[0], num_keypoints, lx * ly)
            locx_pred = locx_pred.reshape(hm_pred.shape[0], num_keypoints, lx * ly)
            locy_pred = locy_pred.reshape(hm_pred.shape[0], num_keypoints, lx * ly)
            _, imax = torch.max(hm_pred, -1)
            likelihood = torch.sigmoid(hm_pred)
            likelihood, _ = torch.max(likelihood, -1)
            # print("likelihood is", likelihood)
            i_y = imax % lx #gives the reminder
            i_x = torch.div(imax, lx, rounding_mode="trunc") # it is the same as imax//lx
            #shapes of locx_mesh, locy_mesh and imax are (8,2)
            x_pred = (locx_pred[locx_mesh, locy_mesh, imax] * (-2 * 3)) + i_x
            y_pred = (locy_pred[locx_mesh, locy_mesh, imax] * (-2 * 3)) + i_y

        x_pred *= 4
        y_pred *= 4
        x_pred_final.extend(x_pred)
        y_pred_final.extend(y_pred)
        likelihood_final.extend(likelihood)

    return x_pred_final,y_pred_final,likelihood_final