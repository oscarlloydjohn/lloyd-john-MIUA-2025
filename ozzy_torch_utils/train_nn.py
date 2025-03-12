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

def train_nn(model_parameters: ModelParameters, train_dataloader: Dataset, test_dataloader: Dataset, mode: str = 'pointnet', unsqueeze: bool = False) -> dict:
    
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
    
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    print(f"Using {device} device")

    model_parameters.model.to(device)

    start_time = datetime.now()

    for epoch in range(model_parameters.num_epochs):
        
        print("------------------------\n\n")
        print(f"Starting epoch {epoch + 1}\n")
        
        # Training loop
        model_parameters.model.train()
        
        running_loss = 0.0

        for batch_idx, dict in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            
            # Access dict returned by dataset __getitem__
            points = dict[model_parameters.data_string]
            labels = dict[model_parameters.labels_string]
            
            # Transpose as in benny script (NB why does it need a transpose)
            if mode == 'pointnet':
                             
                points = points.transpose(2, 1)
            
            # Unsqueeze for resnet
            if unsqueeze:
                
                points = points.unsqueeze(1)
                
                
            
            points, labels = points.to(device), labels.to(device)

            # Forward pass
            logit_output = model_parameters.model(points)[0]

            # Calculate loss, trans_feat argument as None as not used in this function
            if mode == 'pointnet':
                             
                loss = model_parameters.criterion(logit_output, labels, None)
                
            else:
                
                loss = model_parameters.criterion(logit_output, labels)

            # Backpropagation
            loss.backward()
            model_parameters.optimiser.step()
            model_parameters.optimiser.zero_grad()
            
            # Multiply loss by batch size to account for differences in batch size (e.g last batch)
            running_loss += loss.item() * points.size(0)
            
        metrics['training_losses'].append(running_loss/len(train_dataloader))
        
        end_time = datetime.now()
        
        # Validation loop
        model_parameters.model.eval()
        
        # Tensors to store concatenated results from each batch, to compute epoch metrics
        epoch_pred_probs = torch.empty((0, 2))
        epoch_pred_labels = torch.empty(0, dtype=torch.int32)
        epoch_labels = torch.empty(0,dtype=torch.int32)
        
        # Initialise metrics
        running_loss = 0.0
        conf_matrix = BinaryConfusionMatrix()
        accuracy = BinaryAccuracy()
        f1 = BinaryF1Score()
        precision = BinaryPrecision()
        recall = BinaryRecall()
        
        with torch.no_grad():
            
            for batch_idx, dict in enumerate(test_dataloader):
                
                points = dict[model_parameters.data_string]
                labels = dict[model_parameters.labels_string]
                
                # Transpose as in benny script (NB why does it need a transpose)
                if mode == 'pointnet':
                                
                    points = points.transpose(2, 1)
                
                # Unsqueeze for resnet
                if unsqueeze:
                    
                    points = points.unsqueeze(1)
                
                points, labels = points.to(device), labels.to(device)
                
                pred_probability = model_parameters.model(points)[0]
                
                # NB could use lambda function for label conversion
                if mode == 'pointnet':
                
                    running_loss += model_parameters.criterion(pred_probability, labels, None).item() * points.size(0)
                
                    # Apply exponent as the output of the model is log softmax
                    pred_probability = torch.exp(pred_probability)
                    
                    # Threshold is variable to give preference to FN or FP
                    pred_labels = (pred_probability[:, 1] >= model_parameters.threshold).int()
                    
                else:

                    pred_labels = torch.argmax(pred_probability, dim=-1)
                    
                # Accumulate results for metrics 
                epoch_pred_probs = torch.cat((epoch_pred_probs, pred_probability.cpu()))
                epoch_pred_labels = torch.cat((epoch_pred_labels, pred_labels.cpu()))
                epoch_labels = torch.cat((epoch_labels, labels.cpu()))

        end_time = datetime.now()
        
        # Update and compute metrics for each epoch
        [metric.update(epoch_pred_labels, epoch_labels) for metric in [conf_matrix, accuracy, f1, precision, recall]]
        
        [metrics[key].append(metric.compute()) for key, metric in [("conf_matrices", conf_matrix), ("accuracies", accuracy), ("f1s", f1), ("precisions", precision), ("recalls", recall)]]
    
        metrics['roc_curves'].append(roc_curve(epoch_labels, epoch_pred_probs[:, 1], pos_label=1, drop_intermediate=False))
        
        metrics['roc_aucs'].append(roc_auc_score(epoch_labels, epoch_pred_probs[:, 1]))
            
        metrics['validation_losses'].append(running_loss/len(test_dataloader))
        
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
    metrics['num_training_images'] = len(train_dataloader)

    print("Training complete")
    
    return metrics