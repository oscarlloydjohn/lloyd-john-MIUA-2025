"""

Template run
===========

This is an example template of how to use the functions provided in ozzy_torch_utils

:author: Oscar Lloyd-John

"""

'''
IMPORTS
'''

# Torch
import torch
import torch.optim as optim
from torcheval.metrics import *
import torch.nn.functional as F

# Benny pointnet
from pointnet2_benny import pointnet2_cls_msg
from pointnet2_benny import pointnet2_cls_ssg
from pointnet2_benny import provider

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
model_parameters.data_string = 'lhcampus_pointcloud_aligned'

# Dictionary key representing the disease labels
model_parameters.labels_string = 'research_group'

# Lower batch size seemed to give better results
model_parameters.batch_size = 10

# Can drop last batch of the dataset as it will be smaller than the rest
model_parameters.drop_last = False

model_parameters.test_size = 0.2

model_parameters.num_epochs = 200

model_parameters.learning_rate = 0.005

model_parameters.model = 

#Examples:

'''pointnet2_cls_msg.get_model(len(model_parameters.selected_labels), normal_channel=False)'''

'''pointnet2_cls_ssg.get_model(len(model_parameters.selected_labels), normal_channel=False)'''

'''cnn3d_xmuyzz.ResNetV2.generate_model(
            model_depth=18,
            n_classes=2,
            n_input_channels=1,
            shortcut_type='B',
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool=False,
            widen_factor=1.0)'''

model_parameters.optimiser = 

#Example:
'''optim.Adam(
            model_parameters.model.parameters(),
            lr=model_parameters.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4,
            amsgrad=True
            )
'''

'''
Prediction configuration
'''

# Example for pointnet with added noise:
'''def run_prediction(inputs, labels):
    
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
    
    return loss, pred_probability, pred_labels'''

# Example for resnet
'''def run_prediction(inputs, labels):
    
    def run_prediction(inputs, labels):
    
    # Unsqueeze to fit network input (it expects a channel)
    inputs = inputs.unsqueeze(1)
    
    logit_output = model_parameters.model(inputs)
    
    # Assuming BCELoss with logits
    loss = model_parameters.criterion(logit_output[:, 1], labels.to(torch.float))
    
    # Output of last layer is not softmax, so normalise
    pred_probability = F.softmax(logit_output, dim=1)
        
    # Take largest value rather than threshold
    pred_labels = torch.argmax(pred_probability, dim=-1)
    
    return loss, pred_probability, pred_labels

    model_parameters.run_prediction = run_prediction'''

model_parameters.run_prediction = run_prediction


'''
Dataloaders
'''
train_dataloader, test_dataloader = init_dataloaders(model_parameters, verify_data=False, load_in_memory=True)

'''
Loss function configuration
'''

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# Set the criterion after getting the weights
model_parameters.criterion =

# Example:

'''torch.nn.CrossEntropyLoss(weight=(get_weights(train_dataloader)).to(device))'''

'''
Train
'''
metrics, model = train_nn(model_parameters, train_dataloader, test_dataloader, device)

'''
Plot
'''
plot(metrics, model_parameters, save_params=True, save_metrics=True, save_png=True)