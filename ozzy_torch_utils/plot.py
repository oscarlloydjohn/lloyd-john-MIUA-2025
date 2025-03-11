from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime
import pickle

# Custom modules
from preprocessing_post_fastsurfer.subject import *
from preprocessing_post_fastsurfer.vis import *
from ozzy_torch_utils.split_dataset import *
from ozzy_torch_utils.SubjectDataset import *


# NB this function has to remain in the notebook for it to work properly
# Plot training loss, validation loss, and accuracy on separate subplots, along with displaying hyperparameters
def plot(metrics: dict, model_parameters: object, save_params=False, save_metrics=False, save_png=False, ylim=None) -> None:

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

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
    
    ax2.plot(metrics['accuracies'], label='Accuracy', color='green')
    ax2.plot(metrics['f1s'], label='F1 Score', color='blue')
    ax2.plot(metrics['precisions'], label='Precision', color='red')
    ax2.plot(metrics['recalls'], label='Recall', color='orange')
    ax2.set_title('Metrics over epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    info = []
    
    # Metrics
    info.append("Metrics:")
    info.append("\n")
    
    # Training time
    minutes = metrics['train_time'].seconds // 60

    seconds = metrics['train_time'].seconds % 60

    train_time_str = f"Training time: {minutes:02d}m {seconds:02d}s"

    info.append(train_time_str)
    
    # Num images
    try:
        
        info.append(f"Number of training images: {metrics['num_training_images']:.0f}")
        
    except:
        
        print("Error with num_training_images")
        
    info.append("\n")
    
    # Metrics
    info.append("Metrics:")
    info.append("\n")
    info.append(f"Best Accuracy: {max(metrics['accuracies']):.2f}")
    info.append(f"Best F1 Score: {max(metrics['f1s']):.2f}")
    info.append(f"Best Precision: {max(metrics['precisions']):.2f}")
    info.append(f"Best Recall: {max(metrics['recalls']):.2f}")
    info.append(f"Epoch with smallest validation loss: {metrics['validation_losses'].index(min(metrics['validation_losses'])):.0f}")
    info.append("\n\n")
    
    # Parameters
    info.append("Parameters:")
    info.append("\n")
    info.append(f"Model name: {model_parameters.model.__class__.__module__}.{model_parameters.model.__class__.__name__}")
    info.append(f"Model optimiser: {model_parameters.optimiser.__class__.__module__}.{model_parameters.optimiser.__class__.__name__}")
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
    
    name = f"plot_{current_time}"
    
    if save_params:
        
        # Save params as pickle
        with open(f'/uolstore/home/student_lnxhome01/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/figs/{name}_params.pkl', 'wb') as file:
            
            pickle.dump(model_parameters, file)  

    if save_metrics:
        
        # Save metrics as pickle
        with open(f'/uolstore/home/student_lnxhome01/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/figs/{name}_metrics.pkl', 'wb') as file:
            
            pickle.dump(metrics, file)
        
    if save_png:
        
        # Save png as pickle
        plt.savefig(f'/uolstore/home/student_lnxhome01/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/figs/{name}.png', bbox_inches='tight')
    
    plt.show()
    
    return