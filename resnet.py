'''
IMPORTS
'''

# Torch
import torch
import torch.optim as optim
from torcheval.metrics import *
import torch.nn.functional as F

# Benny pointnet
import cnn3d_xmuyzz.C3DNet
from pointnet2_benny import pointnet2_cls_msg
from cnn3d_xmuyzz import ResNetV2
import cnn3d_xmuyzz.cnn

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

model_parameters.data_path = "/uolstore/home/student_lnxhome01/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/scratch-disk/full-datasets/hcampus-large-cohort"

# Disease labels from study
model_parameters.selected_labels = ['CN', 'MCI']

# Dictionary key representing the data of interest
model_parameters.data_string = 'hcampus_vox'

# Dictionary key representing the disease labels
model_parameters.labels_string = 'research_group'

model_parameters.batch_size = 10

model_parameters.test_size = 0.3

model_parameters.num_epochs = 100

model_parameters.learning_rate = 0.001

model_parameters.model = cnn3d_xmuyzz.ResNetV2.generate_model(
                                    model_depth=18,
                                    n_classes=2,
                                    n_input_channels=1,
                                    shortcut_type='B',
                                    conv1_t_size=3,
                                    conv1_t_stride=1,
                                    no_max_pool=True,
                                    widen_factor=1.0)


# Examples
'''pointnet2_cls_msg.get_model(len(model_parameters.selected_labels), normal_channel=False)'''

model_parameters.criterion = torch.nn.BCEWithLogitsLoss(weight=)

# Examples
'''pointnet2_cls_msg.get_loss()'''

'''torch.nn.CrossEntropyLoss()'''

model_parameters.optimiser = optim.Adam(
                                model_parameters.model.parameters(),
                                lr=model_parameters.learning_rate,
                                betas=(0.9, 0.999),
                                eps=1e-08,
                                weight_decay=1e-4,
                                amsgrad=True
                            )

def run_prediction(inputs, labels):
    
    # Unsqueeze to fit network input (it expects a channel)
    inputs = inputs.unsqueeze(1)
    
    logit_output = model_parameters.model(inputs)
    
    loss = model_parameters.criterion(logit_output[:, 1], labels.to(torch.float))
    
    # Output of last layer is not softmax, so normalise
    pred_probability = F.softmax(logit_output, dim=1)
        
    # Take largest value rather than threshold
    pred_labels = torch.argmax(pred_probability, dim=-1)
    
    return loss, pred_probability, pred_labels

model_parameters.run_prediction = run_prediction

# Example prediction func for pointnet
'''def run_prediction(inputs, labels):
    
    # Transpose as in benny script (NB why does it need a transpose)
    inputs = inputs.transpose(2, 1)
    
    logit_output, *_ = model_parameters.model(inputs)
    
    loss = model_parameters.criterion(logit_output, labels, None)
    
    # Apply exponent as the output of the model is log softmax
    pred_probability = torch.exp(logit_output)
        
    # Threshold is variable to give preference to FN or FP
    pred_labels = (pred_probability[:, 1] >= model_parameters.threshold).int()
    
    return loss, pred_probability, pred_labels'''
    
# Example prediction func for resnet/cnn
'''def run_prediction(inputs, labels):
    
    # Unsqueeze to fit network input (it expects a channel)
    inputs = inputs.unsqueeze(1)
    
    logit_output, *_ = model_parameters.model(inputs)
    
    loss = model_parameters.criterion(logit_output, labels)
    
    # Output of last layer is softmax ???
    pred_probability = logit_output
        
    # Take largest value rather than threshold
    pred_labels = torch.argmax(pred_probability, dim=-1)
    
    return loss, pred_probability, pred_labels'''


'''
Dataloaders
'''
train_dataloader, test_dataloader = init_dataloaders(model_parameters, verify_data=False)

'''
Train
'''
metrics = train_nn(model_parameters, train_dataloader, test_dataloader)

'''
Plot
'''
plot(metrics, model_parameters, save_params=True, save_metrics=True, save_png=True)