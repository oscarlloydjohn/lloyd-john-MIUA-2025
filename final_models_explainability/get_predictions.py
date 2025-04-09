"""
Get predictions
===========

Has functions for getting predictions from individual models and also from ensemble. All models take numpy inputs and return numpy outputs. Some functions also return explainability information

:author: Oscar Lloyd-John
"""

import torch
from pointnet2_benny import pointnet2_cls_msg
import pickle
from .explain_pointnet import *
from .explain_volumes import *
from .utils import *
    
def get_pointnet_prediction(input: np.ndarray, device: str) -> tuple:
    """
    Get a prediction from the hippocampus pointnet model for a given input. Takes the input as a numpy array and returns the prediction as a numpy array and the research group as a string. Also returns the attributions of the input features, which can be visualised using the vis_attributions function. Persists the model in memory to avoid loading it multiple times.

    :param input: The input to the model
    :type input: np.ndarray
    :param device: The device to run the model on, for the pipeline this would be cpu
    :type device: str
    :return: The class prediction of the model, the probability output of the model, and the attributions of the input features
    :rtype: tuple
    """

    # Cache model on first call
    if not hasattr(get_pointnet_prediction, "model"):

        model = pointnet2_cls_msg.get_model(2, normal_channel=False)

        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "pointnet.pth"), weights_only=True))

        model.eval()

        model.to(device)

        get_pointnet_prediction.model = model

    else:
        
        model = get_pointnet_prediction.model

    with torch.no_grad():

        input = torch.from_numpy(input).type(torch.float32).to(device)

        attributions = explain_pointnet(model, input)

        # Add batch dim and transpose to 3 x n for pointnet
        input = input.unsqueeze(0).transpose(2, 1)

        # Get output from model, ignoring extra info from pointnet
        output = model(input)[0]

        output = output.squeeze(0)

        # Take negative logits and get probability
        output = torch.nn.functional.softmax(output, dim=0)

        pred_class = int(np.argmax(output.cpu().numpy()))

        output = output.cpu().numpy()

        # Get only the positive class as they add up to 1 anyway
        output = output[1]

        return pred_class, output, attributions

def get_volumes_prediction(input: np.ndarray, struct_names: list = None):

    """

    Get the prediction from the parcellation gradient boosted tree model for a given input. Takes the input as a numpy array and returns the prediction as a numpy array. Also returns the shap values of the input features, which can be visualised using the vis_volumes function or using one of shap's built in plotting methods.

    Note that this is designed to work with the specific normalisation steps in the pipeline. It expects a specific number of regions and a specific order of regions. If the input is not in the correct format, the model will not work as expected.

    :param input: The normalised parcellation volumes input to the model, this is a 1d array of shape (n_structures,)
    :type input: np.ndarray
    :param struct_names: A list of names of the parcellation regions, passed to shap. This will allow regions to be named in the shap plots. If not provided, the regions will be named by their index.
    :type struct_names: list, optional
    :return: The class prediction of the model, the probability output of the model, and the shap values of the input features
    :rtype: tuple
    """

    # Add batch dimension for sklearn
    input = np.expand_dims(input, axis=0)

    # Load model once and cache
    if not hasattr(get_volumes_prediction, "model"):

        with open(os.path.join(os.path.dirname(__file__), "volumes_gbdt.pkl"), 'rb') as file:

            get_volumes_prediction.model = pickle.load(file)

    model = get_volumes_prediction.model

    output = model.predict_proba(input).squeeze(0)[1]
    pred_class = (output >= 0.5).astype(int)
    shap_values = explain_volumes(model, input, struct_names)

    return pred_class, output, shap_values

def get_scores_prediction(input):
    """

    Get the prediction from the neurocognitive test scores gradient boosted tree model for a given input. Takes the input as a numpy array and returns the prediction as a numpy array.

    :param input: The neurocognitive test scores, a list of ints as [mmse, gdscale, faq, npiq]
    :type input: np.ndarray
    :return: The class prediction of the model and the probability output of the model
    :rtype: tuple
    """

    input = np.expand_dims(input, axis=0)

    # Load model once and cache
    if not hasattr(get_scores_prediction, "model"):

        with open(os.path.join(os.path.dirname(__file__), "scores_gbdt.pkl"), 'rb') as file:

            get_scores_prediction.model = pickle.load(file)

    model = get_scores_prediction.model

    output = model.predict_proba(input).squeeze(0)[1]
    pred_class = (output >= 0.5).astype(int)

    return pred_class, output


def get_ensemble_prediction_avg(pointnet_output, volumes_output, scores_output, scores=True):

    if scores:

        average = (pointnet_output + volumes_output + scores_output)/3
    
    else:

        average = (pointnet_output + volumes_output)/2

    pred_class = round(average)

    return pred_class, average

# Max probability rule (Lassila et. al)
# Need to verify this is correct
def get_ensemble_prediction_maxprob(pointnet_tuple, volumes_tuple, scores_tuple, scores=True):

    max_confidence = -np.inf

    best_pred = None

    best_index = -1

    outputs_list = [pointnet_tuple, volumes_tuple, scores_tuple]

    if scores == False:
    
        outputs_list.pop()

    for i, pred in enumerate(outputs_list):

        if pred[1] > max_confidence:

            max_confidence = pred[1]

            best_pred = pred

            best_index = i

    return best_pred[0], best_pred[1], best_index