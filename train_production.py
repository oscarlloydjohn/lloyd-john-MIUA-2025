"""

Train production
===========

This is the final parameter configuration for training the production model on train and validation sets, leaving the holdout

:author: Oscar Lloyd-John

"""
# Torch
import torch
import torch.optim as optim
from torcheval.metrics import *
import torch.nn.functional as F

# Benny pointnet
from pointnet2_benny import pointnet2_cls_msg
from pointnet2_benny import provider
import cnn3d_xmuyzz.ResNetV2

# Custom modules
from preprocessing_post_fastsurfer.subject import *
from preprocessing_post_fastsurfer.vis import *
from ozzy_torch_utils.split_dataset import *
from ozzy_torch_utils.subject_dataset import *
from ozzy_torch_utils.plot import *
from ozzy_torch_utils.train_nn import *
from ozzy_torch_utils.model_parameters import *
from ozzy_torch_utils.init_dataloaders import *

import torch
from torcheval.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_curve, roc_auc_score
from ozzy_torch_utils.model_parameters import *

model_parameters = ModelParameters()

model_parameters.data_path = "/uolstore/home/student_lnxhome01/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/scratch-disk/full-datasets/hcampus-1.5T-cohort"

# Disease labels from study
model_parameters.selected_labels = ['CN', 'MCI']

# Dictionary key representing the data of interest
model_parameters.data_string = 'lhcampus_pointcloud_aligned'

# Dictionary key representing the disease labels
model_parameters.labels_string = 'research_group'

# Lower batch size seemed to give better results
model_parameters.batch_size = 10

# Can drop last batch of the dataset as it will be smaller than the rest
model_parameters.drop_last = False

model_parameters.test_size = 0.2

model_parameters.num_epochs = 180

model_parameters.learning_rate = 0.005

model_parameters.model = pointnet2_cls_msg.get_model(len(model_parameters.selected_labels), normal_channel=False)

model_parameters.optimiser = optim.Adam(
                                model_parameters.model.parameters(),
                                lr=model_parameters.learning_rate,
                                betas=(0.9, 0.999),
                                eps=1e-08,
                                weight_decay=1e-4,
                                amsgrad=True
                            )

model_parameters.scheduler = torch.optim.lr_scheduler.LinearLR(model_parameters.optimiser, start_factor=1.0, end_factor=0.01, total_iters=50)

'''
Prediction configuration
'''
def run_prediction(inputs, labels):
    
    inputs = inputs.cpu().data.numpy()
    inputs = provider.random_point_dropout(inputs)
    inputs[:, :, 0:3] = provider.shift_point_cloud(inputs[:, :, 0:3])
    inputs = torch.Tensor(inputs).cuda()
    inputs = inputs.transpose(2, 1)
    
    logit_output, *_ = model_parameters.model(inputs)
    
    loss = model_parameters.criterion(logit_output, labels)
    
    # Apply exponent as the output of the model is log softmax
    pred_probability = F.softmax(logit_output, dim=1)
        
    pred_labels = torch.argmax(pred_probability, dim=1)
    
    return loss, pred_probability, pred_labels

model_parameters.run_prediction = run_prediction

dataset = SubjectDataset(model_parameters.data_path, model_parameters.selected_labels, load_in_memory = True, data_string=model_parameters.data_string, labels_string=model_parameters.labels_string)

dataloader = DataLoader(dataset, batch_size=model_parameters.batch_size, shuffle=True, num_workers = 4, drop_last=model_parameters.drop_last)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# Set the criterion after getting the weights
model_parameters.criterion = torch.nn.CrossEntropyLoss(weight=(get_weights(dataloader)).to(device))

training_losses, model = train_nn(model_parameters, dataloader, None, device, train_only=True, scheduler_start=50)

plt.plot(training_losses)
plt.show()

model.eval()

model.to('cpu')

torch.save(model.state_dict(), 'pointnet.pth')




