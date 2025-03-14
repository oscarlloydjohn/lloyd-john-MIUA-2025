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
model_parameters.data_string = 'brain'

# Dictionary key representing the disease labels
model_parameters.labels_string = 'research_group'

# Lower batch size seemed to give better results
model_parameters.batch_size = 5

# Can drop last batch of the dataset as it will be smaller than the rest
model_parameters.drop_last = False

model_parameters.test_size = 0.2

model_parameters.num_epochs = 150

model_parameters.learning_rate = 0.001

model_parameters.model = cnn3d_xmuyzz.ResNetV2.generate_model(
                            model_depth=10,
                            n_classes=2,
                            n_input_channels=1,
                            shortcut_type='B',
                            conv1_t_size=7,
                            conv1_t_stride=1,
                            no_max_pool=False,
                            widen_factor=1.0)

model_parameters.optimiser = optim.Adam(
                                model_parameters.model.parameters(),
                                lr=model_parameters.learning_rate,
                                betas=(0.9, 0.999),
                                eps=1e-08,
                                weight_decay=1e-4,
                                amsgrad=True
                            )

'''
Prediction configuration
'''
def run_prediction(inputs, labels):
    
    inputs = inputs.unsqueeze(1)
    
    logit_output, *_ = model_parameters.model(inputs)
    
    loss = model_parameters.criterion(logit_output, labels)
    
    logit_output = model_parameters.model(inputs)
    
    # Assuming CE loss
    loss = model_parameters.criterion(logit_output, labels)
    
    # Output of last layer is not softmax, so normalise
    pred_probability = F.softmax(logit_output, dim=1)
        
    # Take largest value rather than threshold
    pred_labels = torch.argmax(pred_probability, dim=-1)
    
    return loss, pred_probability, pred_labels

model_parameters.run_prediction = run_prediction


'''
Dataloaders
'''
train_dataloader, test_dataloader = init_dataloaders(model_parameters, verify_data=False)

'''
Loss function configuration
'''

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# Set the criterion after getting the weights
model_parameters.criterion = torch.nn.CrossEntropyLoss(weight=(get_weights(train_dataloader)).to(device))

'''
Train
'''
metrics = train_nn(model_parameters, train_dataloader, test_dataloader, device)

'''
Plot
'''
plot(metrics, model_parameters, save_params=True, save_metrics=True, save_png=True)