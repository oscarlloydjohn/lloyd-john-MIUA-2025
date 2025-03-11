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
from ozzy_torch_utils.split_dataset import *
from ozzy_torch_utils.SubjectDataset import *

def train_nn(model_parameters, train_dataloader, test_dataloader):
    
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
        
        print(f"Starting epoch {epoch + 1}\n")
        
        # Training loop
        model_parameters.model.train()
        
        running_loss = 0.0

        for batch_idx, dict in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            
            # Access dict returned by dataset __getitem__
            points = dict[model_parameters.data_string]
            labels = dict[model_parameters.labels_string]
            
            # Transpose as in benny script (NB why does it need a transpose)
            points = points.transpose(2, 1)
            
            points, labels = points.to(device), labels.to(device)

            # Forward pass
            output, _ = model_parameters.model(points)                

            # Calculate loss, trans_feat argument as None as not used in this function
            if "pointnet" in f"{model_parameters.model.__class__.__module__}.{model_parameters.model.__class__.__name__}":
                             
                loss = model_parameters.criterion(output, labels, None)
                
            else:
                
                loss = model_parameters.criterion(output, labels)

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
        
        # Initialise loop metrics
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
                
                points = points.transpose(2, 1)
                
                points, labels = points.to(device), labels.to(device)
                
                pred_probability, _ = model_parameters.model(points)
                
                # NB could use lambda function for label conversion
                if "pointnet" in f"{model_parameters.model.__class__.__module__}.{model_parameters.model.__class__.__name__}":
                
                    running_loss += model_parameters.criterion(output, labels, None).item() * points.size(0)
                
                    # Apply exponent as the output of the model is log softmax
                    pred_probability = torch.exp(pred_probability)
                    
                    # Threshold is variable to give preference to FN or FP
                    pred_labels = (pred_probability[:, 1] >= model_parameters.threshold).int()
                    
                else:

                    pred_labels = torch.argmax(pred_probability, dim=-1)

                # Update metrics
                [metric.update(pred_labels, labels) for metric in [conf_matrix, accuracy, f1, precision, recall]]
                
                metrics["roc_curves"].append(roc_curve(labels, pred_probability, pos_label=1))
                
                metrics["roc_aucs"].append(roc_auc_score(labels, pred_probability))

        end_time = datetime.now()
                
        # Append metric lists
        [metrics[key].append(metric.compute()) for key, metric in [("conf_matrices", conf_matrix), ("accuracies", accuracy), ("f1s", f1), ("precisions", precision), ("recalls", recall)]]       
            
        metrics['validation_losses'].append(running_loss/len(test_dataloader))
        
        print(f"\nEpoch {epoch + 1} complete\n")
        print("------------------------")
        print(conf_matrix.compute())
        print(f"Training Loss:   {metrics['training_losses'][-1]:.4f}")
        print(f"Validation Loss: {metrics['validation_losses'][-1]:.4f}")
        print(f"Accuracy:        {metrics['accuracies'][-1]:.4f}")
        print(f"F1 Score:        {metrics['f1s'][-1]:.4f}")
        print(f"Precision:       {metrics['precisions'][-1]:.4f}")
        print(f"Recall:          {metrics['recalls'][-1]:.4f}")
        print(f"ROC AUC:         {metrics['roc_aucs'][-1]:.4f}")
        print("------------------------\n\n")
            
        # Break before nightly restart
        current_time = datetime.now()
        
        if current_time.hour == 23 and current_time.minute >= 30:
            
            print("Break before nightly restart")
            
            break
        
    metrics['train_time'] = end_time - start_time
    metrics['num_training_images'] = len(train_dataloader)

    torch.save(model_parameters.model.state_dict(), 'trained_model.pth')

    print("Training complete and model saved")
    
    return metrics