
# Torch
import torch
import torch.optim as optim
from torcheval.metrics import *
import torch.nn.functional as F

# Benny pointnet
from pointnet2_benny import pointnet2_cls_ssg
import shutil

# Custom modules
from preprocessing_pre_fastsufer.preprocess import *
from preprocessing_post_fastsurfer.subject import *
from preprocessing_post_fastsurfer.vis import *
from ozzy_torch_utils.split_dataset import *
from ozzy_torch_utils.subject_dataset import *
from ozzy_torch_utils.plot import *
from ozzy_torch_utils.train_nn import *
from ozzy_torch_utils.model_parameters import *
from ozzy_torch_utils.init_dataloaders import *

from explain_pointnet import get_prediction

# Set seed for NN as it seems to behave differently each time
def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

# Use pointnet model to run a prediction on a numpy input, and return a string of the research group along with the output as a numpy array
def get_pointnet_prediction(input, device, seed=42, mode = 'pth'):

    set_seed(seed)

    with torch.no_grad():

        if mode == 'pth':

            model = pointnet2_cls_ssg.get_model(2, normal_channel=False)

            model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "pointnet.pth"), weights_only=True))

        if mode == 'pkl':

            pkl_path = "/uolstore/home/student_lnxhome01/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/runs/run_21-03-2025_15-28-42/run_21-03-2025_15-28-42_params.pkl"

            with open(pkl_path, 'rb') as file:
        
                params = pickle.load(file)

            model = params.model

        model.eval()

        input = torch.from_numpy(input).type(torch.float32).to(device)

        # Add batch dim and transpose to 3 x n for pointnet
        input = input.unsqueeze(0).transpose(2,1)

        model.to(device)

        # Get output from model, ignoring extra info from pointnet
        output = model(input)[0]

        output = output.squeeze(0)

        # Take negative logits and get probability
        output = torch.nn.functional.softmax(output, dim=0)

        pred_class = int(np.argmax(output.cpu().numpy()))

        mapping = {
            0: 'CN',
            1: 'MCI',
        }

        pred_research_group = mapping.get(pred_class, -1)

        output = output.cpu().numpy()

        # Get only the positive class as they add up to 1 anyway
        output = output[1]

        return pred_research_group, pred_class, output

def get_volumes_prediction(input):

    # Sklearn expects batch dim
    input = [input]

    with open(os.path.join(os.path.dirname(__file__), "volumes_gbdt.pkl"), 'rb') as file:
    
        model = pickle.load(file)

    output = model.predict_proba(input).squeeze(0)[1]

    pred_class = int(model.predict(input))

    mapping = {
            0: 'CN',
            1: 'MCI',
        }

    pred_research_group = mapping.get(pred_class, -1)

    return pred_research_group, pred_class, output

def get_scores_prediction(input):

    input = [input]

    with open(os.path.join(os.path.dirname(__file__), "scores_gbdt.pkl"), 'rb') as file:
    
        model = pickle.load(file)

    output = model.predict_proba(input).squeeze(0)[1]

    pred_class = int(model.predict(input))

    mapping = {
            0: 'CN',
            1: 'MCI',
        }

    pred_research_group = mapping.get(pred_class, -1)

    return pred_research_group, pred_class, output

def get_ensemble_prediction_avg(subject_data):

    _, _, pointnet_output = get_pointnet_prediction(subject_data['lhcampus_pointcloud_aligned'], 'cpu')

    _, _, volumes_output = get_volumes_prediction(subject_data['volumes'])

    if subject_data['scores'] is not None:
        
        _, _, scores_output = get_scores_prediction(subject_data['scores'])

        average = (pointnet_output + volumes_output + scores_output)/3
    
    else:

        average = (pointnet_output + volumes_output)/2

    pred_class = round(average)

    mapping = {
        0: 'CN',
        1: 'MCI',
    }

    pred_research_group = mapping.get(pred_class, -1)

    return pred_research_group, pred_class, average

'''# Max probability rule (Lassila et. al)
# Need to verify this is correct
def get_ensemble_prediction_maxprob(subject_data):

    pointnet_pred = get_pointnet_prediction(subject_data['lhcampus_pointcloud_aligned'], 'cpu')
    
    volumes_pred = get_volumes_prediction(subject_data['volumes'])

    scores_pred = get_scores_prediction(subject_data['scores'])

    max_confidence = -np.inf

    best_pred = None

    best_index = -1

    for i, pred in enumerate([pointnet_pred, volumes_pred, scores_pred]):

        if pred[2] > max_confidence:

            max_confidence = pred[2]

            best_pred = pred

            best_index = i

    return best_pred[0], best_pred[1], best_pred[2], best_index'''

# Should it be max for only the positive class or for both classes?
def get_ensemble_prediction_maxprob(subject_data):

    pointnet_pred = get_pointnet_prediction(subject_data['lhcampus_pointcloud_aligned'], 'cpu')

    volumes_pred = get_volumes_prediction(subject_data['volumes'])

    scores_pred = get_scores_prediction(subject_data['scores'])

    max_confidence = -np.inf

    best_pred = None

    best_index = -1

    for i, pred in enumerate([pointnet_pred, volumes_pred, scores_pred]):

        # Consider both high positive and high negative confidence
        confidence = max(pred[2], 1 - pred[2])  

        if confidence > max_confidence:

            max_confidence = confidence

            best_pred = pred

            best_index = i

    return best_pred[0], best_pred[1], best_pred[2], best_index