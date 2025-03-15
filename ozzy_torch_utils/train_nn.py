# Torch
import torch
import torch
from torcheval.metrics import *

# Other
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_curve, roc_auc_score

# Custom modules
from preprocessing_post_fastsurfer.subject import *
from preprocessing_post_fastsurfer.vis import *
from .split_dataset import *
from .subject_dataset import *
from .model_parameters import *

def train_nn(model_parameters: ModelParameters, train_dataloader: Dataset, test_dataloader: Dataset, device) -> dict:
    
    for attr, value in vars(model_parameters).items():
        
        if value is None:
            
            print(f"Error: model parameter {attr} incomplete")
            
            return
    
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

    print(f"Using {device} device")

    model_parameters.model.to(device)

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
            
            # Function as defined for given model
            loss, pred_probability, pred_labels = model_parameters.run_prediction(inputs, labels)

            # Backpropagation
            loss.backward()
            model_parameters.optimiser.step()
            model_parameters.optimiser.zero_grad()
            
            # Multiply loss by batch size to account for differences in batch size (e.g last batch)
            running_training_loss += loss.item() * inputs.size(0)
        
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
        
        metrics['training_losses'].append(running_training_loss/len(train_dataloader))
            
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
            
        # Break before nightly restart
        current_time = datetime.now()
        
        if current_time.hour == 23 and current_time.minute >= 30:
            
            print("Break before nightly restart")
            
            break
        
    metrics['train_time'] = end_time - start_time

    # THis is wrong, len dataloader is the number of batches
    # metrics['num_training_images'] = len(train_dataloader)

    # metrics['num_training_images'] = len(train_dataloader.dataset)

    print("Training complete")
    
    return metrics