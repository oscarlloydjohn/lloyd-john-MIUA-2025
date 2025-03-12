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
from ozzy_torch_utils.SubjectDataset import *
from ozzy_torch_utils.plot import *
from ozzy_torch_utils.train_nn import *
from ozzy_torch_utils.model_parameters import *
from ozzy_torch_utils.init_dataloaders import *

'''
Params
'''

model_parameters = ModelParameters()

model_parameters.data_path = "/uolstore/home/student_lnxhome01/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/scratch-disk/full-datasets/adni1-complete-3T-processed"

# Disease labels from study
model_parameters.selected_labels = ['CN', 'MCI']

# Dictionary key representing the data of interest
model_parameters.data_string = 

# Dictionary key representing the disease labels
model_parameters.labels_string = 'research_group'

# Prevent class imbalance
model_parameters.downsample_majority = True

# NB this argument makes prevent_id_leakage redundant
model_parameters.single_img_per_subject = False

# Prevent the same subject id from occuring in train and test, in case of more than one image per id
model_parameters.prevent_id_leakage = True

model_parameters.batch_size = 20

model_parameters.test_size = 0.3

model_parameters.num_epochs = 200

model_parameters.learning_rate = 0.001

model_parameters.threshold = 0.5

model_parameters.model = 

# Examples
'''pointnet2_cls_msg.get_model(len(model_parameters.selected_labels), normal_channel=False)'''

'''cnn3d_xmuyzz.ResNetV2.generate_model(
            model_depth=18,
            n_classes=2,
            n_input_channels=1,
            shortcut_type='B',
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool=False,
            widen_factor=1.0)'''

model_parameters.criterion = 

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

'''
Dataloaders
'''
train_dataloader, test_dataloader = init_dataloaders(model_parameters, verify_data=False)

'''
Train
'''
metrics = train_nn(model_parameters, train_dataloader, test_dataloader, mode=, unsqueeze=)

'''
Plot
'''
plot(metrics, model_parameters, save_params=True, save_metrics=True, save_png=True)