'''
IMPORTS
'''

# Torch
import torch
import torch.optim as optim
from torcheval.metrics import *

# Benny pointnet
from pointnet2_benny import pointnet2_cls_msg
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

'''
Params
'''

model_parameters = ModelParameters()

model_parameters.data_path = "/uolstore/home/student_lnxhome01/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/scratch-disk/full-datasets/hcampus-1.5T-cohort"

# Disease labels from study
model_parameters.selected_labels = ['CN', 'MCI']

# Dictionary key representing the data of interest
model_parameters.data_string = 'hcampus_pointcloud'

# Dictionary key representing the disease labels
model_parameters.labels_string = 'research_group'

# Prevent class imbalance
model_parameters.downsample_majority = True

# Lower batch size seemed to give better results
model_parameters.batch_size = 5

model_parameters.test_size = 0.3

model_parameters.num_epochs = 5

model_parameters.learning_rate = 0.001

model_parameters.model = pointnet2_cls_msg.get_model(len(model_parameters.selected_labels), normal_channel=False)

model_parameters.criterion = pointnet2_cls_msg.get_loss()

model_parameters.optimiser = optim.Adam(
                                model_parameters.model.parameters(),
                                lr=model_parameters.learning_rate,
                                betas=(0.9, 0.999),
                                eps=1e-08,
                                weight_decay=1e-4,
                                amsgrad=True
                            )

def run_prediction(inputs, labels):
    
    # Transpose as in benny script (NB why does it need a transpose)
    inputs = inputs.transpose(2, 1)
    
    logit_output, *_ = model_parameters.model(inputs)
    
    loss = model_parameters.criterion(logit_output, labels, None)
    
    # Apply exponent as the output of the model is log softmax
    pred_probability = torch.exp(logit_output)
        
    # Threshold is variable to give preference to FN or FP
    pred_labels = (pred_probability[:, 1] >= 0.5).int()
    
    return loss, pred_probability, pred_labels

model_parameters.run_prediction = run_prediction
    

'''
Dataloaders
'''
train_dataloader, test_dataloader = init_dataloaders(model_parameters, verify_data=False)

'''
Train
'''
metrics = train_nn(model_parameters, train_dataloader, test_dataloader, mode='pointnet', unsqueeze=False)

'''
Plot
'''
plot(metrics, model_parameters, save_params=True, save_metrics=True, save_png=True)