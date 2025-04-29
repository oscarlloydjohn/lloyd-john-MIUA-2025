"""

Train neural network
===========

This module provides a function that trains a neural network classifier using the given model parameters (ModelParameters) and dataset. The function returns a dictionary of metrics for each epoch and the trained model. It is desiged to be used with the ModelParameters class, and the plot function also contained in this package. The plot function can be used to visualise the metrics returned by this function, and also to save them.

:author: Oscar Lloyd-John

"""

import torch
from torcheval.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_curve, roc_auc_score
from .model_parameters import *

def train_nn(model_parameters: ModelParameters, train_dataloader: DataLoader, test_dataloader: DataLoader, device: torch.device, train_only: bool = False, scheduler_start: int = 0) -> tuple[dict, torch.nn.Module]:
    
    """

    Trains a neural network using the given model parameters and dataset. The function returns a dictionary of metrics for each epoch and the trained model. Also can train a model without evaluation i.e for a production model to be created. 

    :param model_parameters: The model parameters to be used for training
    :type model_parameters: ModelParameters
    :param train_dataloader: The dataloader for the training dataset
    :type train_dataloader: DataLoader
    :param test_dataloader: The dataloader for the test dataset
    :type test_dataloader: DataLoader
    :param device: The device to train the model on
    :type device: torch.device
    :param train_only: If True, the function will train the model without evaluating it (meaning it only uses one dataset). Returns the training losses and the trained model
    :type train_only: bool
    :param scheduler_start: The epoch to start the scheduler on, if there is one
    :type scheduler_start: int
    :return: A dictionary of metrics stored as lists where a list contains all metrics of one type over an epoch and the trained model
    :rtype: tuple[dict, torch.nn.Module]

    """

    # Check that all ModelParameters attributes were set
    for attr, value in vars(model_parameters).items():
        
        if value is None:
            
            print(f"Error: model parameter {attr} incomplete")
            
            return
    
    # Each list in the dictionary will contain the metrics for each epoch
    metrics = {
        "training_losses" : [],
        "validation_losses": [],
        "conf_matrices": [],
        "accuracies": [],
        "f1s": [],
        "precisions": [],
        "recalls": [],
        "roc_curves": [],
        "roc_aucs": [],
        "train_time": None,
        "num_training_images": None
    }

    # Device setup
    print(f"Using {device} device")

    model_parameters.model.to(device)

    # Monitor training time
    start_time = datetime.now()

    for epoch in range(model_parameters.num_epochs):
        
        print("------------------------\n\n")
        print(f"Starting epoch {epoch + 1}\n")
        
        # Training loop
        model_parameters.model.train()
        
        running_training_loss = 0.0
        
        print(f"\nTraining:")
        for batch_idx, dict in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            
            # Access dict returned by dataset __getitem__
            inputs = dict[model_parameters.data_string]
            labels = dict[model_parameters.labels_string]
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Run_prediction is a function that handles running the model and calculating the loss and predicted classes
            loss, pred_probability, pred_labels = model_parameters.run_prediction(inputs, labels)

            # Backpropagation
            loss.backward()
            model_parameters.optimiser.step()
            model_parameters.optimiser.zero_grad()
            
            # Multiply loss by batch size to account for differences in batch size (e.g last batch)
            running_training_loss += loss.item() * inputs.size(0)

        # If the instance has a scheduler added
        if hasattr(model_parameters, 'scheduler'):

            if epoch >= scheduler_start:

                model_parameters.scheduler.step()

        metrics['training_losses'].append(running_training_loss/len(train_dataloader))

        if train_only:

            continue
        
        # Validation loop
        model_parameters.model.eval()
        
        # Tensors to store concatenated results from each batch, to compute epoch metrics
        epoch_pred_probs = torch.empty((0, 2), device=device)
        epoch_pred_labels = torch.empty(0, dtype=torch.int32, device=device)
        epoch_labels = torch.empty(0,dtype=torch.int32, device=device)
        
        # Initialise metrics
        running_validation_loss = 0.0
        conf_matrix = BinaryConfusionMatrix()
        accuracy = BinaryAccuracy()
        f1 = BinaryF1Score()
        precision = BinaryPrecision()
        recall = BinaryRecall()
        
        with torch.no_grad():
            
            print(f"\nTesting:")
            for batch_idx, dict in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                
                inputs = dict[model_parameters.data_string]
                labels = dict[model_parameters.labels_string]
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                loss, pred_probability, pred_labels = model_parameters.run_prediction(inputs, labels)
                    
                running_validation_loss += loss.item() * inputs.size(0)
                    
                # Accumulate results for metrics
                epoch_pred_probs = torch.cat((epoch_pred_probs, pred_probability))
                epoch_pred_labels = torch.cat((epoch_pred_labels, pred_labels))
                epoch_labels = torch.cat((epoch_labels, labels))

        end_time = datetime.now()
        
        # Update and compute metrics for each epoch
        for metric in [conf_matrix, accuracy, f1, precision, recall]:
            
            metric.update(epoch_pred_labels, epoch_labels) 
         
        for key, metric in [("conf_matrices", conf_matrix), ("accuracies", accuracy), ("f1s", f1), ("precisions", precision), ("recalls", recall)]:
            
            metrics[key].append(metric.compute()) 
    
        metrics['roc_curves'].append(roc_curve(epoch_labels.cpu(), epoch_pred_probs[:, 1].cpu(), pos_label=1, drop_intermediate=False))
        
        metrics['roc_aucs'].append(roc_auc_score(epoch_labels.cpu(), epoch_pred_probs[:, 1].cpu()))
            
        metrics['validation_losses'].append(running_validation_loss/len(test_dataloader))
        
        # Reset torcheval metrics for next epoch
        [metric.reset() for metric in [conf_matrix, accuracy, f1, precision, recall]]
        
        print(f"\nEpoch {epoch + 1} complete\n")
        print(f"Training Loss:   {metrics['training_losses'][-1]:.4f}")
        print(f"Validation Loss: {metrics['validation_losses'][-1]:.4f}")
        print(f"Accuracy:        {metrics['accuracies'][-1]:.4f}")
        print(f"F1 Score:        {metrics['f1s'][-1]:.4f}")
        print(f"Precision:       {metrics['precisions'][-1]:.4f}")
        print(f"Recall:          {metrics['recalls'][-1]:.4f}")
        print(f"ROC AUC:         {metrics['roc_aucs'][-1]:.4f}")
            
        # Break before nightly restart on feng-linux machines
        current_time = datetime.now()
        
        if current_time.hour == 23 and current_time.minute >= 30:
            
            print("Break before nightly restart")
            
            break
        
    # For final model production
    if train_only:

        return metrics['training_losses'], model_parameters.model
        
    metrics['train_time'] = end_time - start_time

    metrics['num_training_images'] = len(train_dataloader.dataset)

    print("Training complete")
    
    return metrics, model_parameters.model