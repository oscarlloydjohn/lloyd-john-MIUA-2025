"""

Plot
===========

This module provides a way to plot and store the results of a model training. 

:author: Oscar Lloyd-John

"""

from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime
import pickle
from .subject_dataset import *
from .model_parameters import *

# NB this function has to remain in the notebook for it to work properly
# Plot training loss, validation loss, and accuracy on separate subplots, along with displaying hyperparameters
def plot(metrics: dict, model_parameters: ModelParameters, save_params: bool = False, save_metrics: bool = False, save_png: bool = False, ylim: tuple = None) -> None:

    """
    
    This function plots the training and validation loss, accuracy, F1 score, precision, recall, and ROC AUC over epochs. It also displays the hyperparameters used for training as given by the ModelParameters object. The function can save the hyperparameters (including the trained model itself), metrics, and plot as a pickle file and png respectively. These can then be recalled and used again in this function.

    :param metrics: The metrics dictionary returned by the train_nn function
    :type metrics: dict
    :param model_parameters: The model parameters used for training
    :type model_parameters: ModelParameters
    :param save_params: Whether to save the model parameters (including the model itself) as a pickle file
    :type save_params: bool
    :param save_metrics: Whether to save the metrics dict as a pickle file
    :type save_metrics: bool
    :param save_png: Whether to save the plot as a png file
    :type save_png: bool
    :param ylim: The y-axis limits for the training and validation loss plot
    :type ylim: tuple
    :return: None

    """

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=False)

    # Plot for Training and Validation Loss
    ax1.plot(metrics['training_losses'], label='Training Loss', color='blue')
    ax1.plot(metrics['validation_losses'], label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss over Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    if ylim is not None:
        ax1.set_ylim(ylim)
    else:
        ax1.set_ylim(0, metrics['training_losses'][0] + 5)

    ax1.grid(True)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot for Metrics over Epochs
    ax2.plot(metrics['accuracies'], label='Accuracy', color='green')
    ax2.plot(metrics['f1s'], label='F1 Score', color='blue')
    ax2.plot(metrics['precisions'], label='Precision', color='red')
    ax2.plot(metrics['recalls'], label='Recall', color='orange')
    
    print(len(metrics['roc_aucs']))
    
    # ROCs were sometimes nan when model performed poorly i.e always predicted one class
    if True not in np.isnan(metrics['roc_aucs']):
        
        ax2.plot(metrics['roc_aucs'], label='ROC AUCs', color='purple')
        
    ax2.set_title('Metrics over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot for ROC Curve
    fpr, tpr, thresholds = metrics['roc_curves'][-1]
    ax3.plot(fpr, tpr, label='AUC', color='purple')
    ax3.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax3.set_title('ROC Curve from final epoch')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_xlim(0,1)
    ax3.legend()
    ax3.grid(True)

    ax3.xaxis.set_major_locator(MaxNLocator(integer=False))

    info = []
    
    # Metrics
    info.append("Metrics:")
    info.append("\n")
    
    # Training time
    minutes = metrics['train_time'].seconds // 60

    seconds = metrics['train_time'].seconds % 60

    train_time_str = f"Training time: {minutes:02d}m {seconds:02d}s"

    info.append(train_time_str)
    
    # Num images sometimes failed when plotting from pickle
    try:
        
        info.append(f"Number of training images: {metrics['num_training_images']:.0f}")
        
    except:
        
        print("Error with num_training_images")
        
    info.append("\n")
    
    
    '''# Metrics
    info.append(f"Best Accuracy: {max(metrics['accuracies']):.2f}")
    info.append(f"Best F1 Score: {max(metrics['f1s']):.2f}")
    info.append(f"Best Precision: {max(metrics['precisions']):.2f}")
    info.append(f"Best Recall: {max(metrics['recalls']):.2f}")
    info.append(f"Best ROC AUC: {max(metrics['roc_aucs']):.2f}")
    info.append(f"Epoch with smallest validation loss: {metrics['validation_losses'].index(min(metrics['validation_losses'])):.0f}")
    info.append("\n\n")'''
    
    info.append(f"Final Accuracy: {metrics['accuracies'][-1]:.2f}")
    info.append(f"Final F1 Score: {metrics['f1s'][-1]:.2f}")
    info.append(f"Final Precision: {metrics['precisions'][-1]:.2f}")
    info.append(f"Final Recall: {metrics['recalls'][-1]:.2f}")
    info.append(f"Final ROC AUC: {metrics['roc_aucs'][-1]:.2f}")
    info.append(f"Epoch with smallest validation loss: {metrics['validation_losses'].index(min(metrics['validation_losses'])):.0f}")
    info.append("\n\n")
    
    
    # Parameters
    info.append("Parameters:")
    info.append("\n")
    info.append(f"Model name: {model_parameters.model.__class__.__module__}.{model_parameters.model.__class__.__name__}")
    info.append(f"Model optimiser: {model_parameters.optimiser}")
    info.append(f"Model criterion: {model_parameters.criterion.__class__.__module__}.{model_parameters.criterion.__class__.__name__}")

    for name, value in model_parameters.__dict__.items():
        
        if name == "model" or name == "optimiser" or name == "criterion":
            
            continue
        
        info.append(f"{name}: {str(value)[-50:]}")
    
    info_text = "\n".join(info)
    
    # Add text to fig
    fig.text(0.5, 0.02, info_text, ha='center', va='top', wrap=True, fontsize=10)
    
    # Calculate time for filename
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    print(f"Finish time: {current_time}")
    
    if save_params or save_metrics or save_png:
        
        name = f"run_{current_time}"
            
        directory = f'/uolstore/home/student_lnxhome01/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/runs/{name}'
        
        os.makedirs(directory, exist_ok=True)

    if save_png:
        
        # Save png as pickle
        plt.savefig(f'{directory}/{name}.png', bbox_inches='tight')

    if save_metrics:
        
        # Save metrics as pickle
        with open(f'{directory}/{name}_metrics.pkl', 'wb') as file:
            
            pickle.dump(metrics, file)
    
    # Save params not tested yet
    if save_params:
        
        # run_prediction will not be available on load, it must be redefined
        model_parameters.run_prediction = None

        # Move model to cpu so it can be loaded on devices without GPU
        model_parameters.model.cpu()
        
        # Save params as pickle
        with open(f'{directory}/{name}_params.pkl', 'wb') as file:
            
            pickle.dump(model_parameters, file)  

    plt.show()

    return