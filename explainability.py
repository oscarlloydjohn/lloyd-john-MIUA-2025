# %%
# Torch
import torch
import torch.optim as optim
from torcheval.metrics import *

# Benny pointnet
from pointnet2_benny import pointnet2_cls_msg
import cnn3d_xmuyzz.ResNetV2

import dill as pickle
from captum.attr import *
import pyvista as pv
from matplotlib.colors import Normalize

# Custom modules
from preprocessing_post_fastsurfer.subject import *
from preprocessing_post_fastsurfer.vis import *
from ozzy_torch_utils.split_dataset import *
from ozzy_torch_utils.subject_dataset import *
from ozzy_torch_utils.plot import *
from ozzy_torch_utils.train_nn import *
from ozzy_torch_utils.model_parameters import *
from ozzy_torch_utils.init_dataloaders import *

# %%
pickle_pathname = "/uolstore/home/student_lnxhome01/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/runs/run_17-03-2025_19-41-28/run_17-03-2025_19-41-28_params.pkl"

with open(pickle_pathname, 'rb') as file:
    
    model_parameters = pickle.load(file)
    
model = model_parameters.model

# %%
model.eval()

with torch.no_grad():

    # Wrap model as pointnet_cls outputs a tuple for some reason
    wrapped_model = lambda x: model(x)[0]

    ig = IntegratedGradients(wrapped_model)
    
    # Example data
    cloud = np.load("/uolstore/home/student_lnxhome01/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/scratch-disk/full-datasets/hcampus-1.5T-cohort/ADNI_002_S_0413_MR_MPR-R__GradWarp__B1_Correction_Br_20070319120315662_S13894_I45122/Left-Hippocampus_aligned_cropped_mesh_downsampledcloud.npy")

    input = torch.from_numpy(cloud)

    # NN expects float32 on cuda
    input = input.type(torch.float32).to('cuda')

    # Unsqueeze to add empty batch dimension then transpose  to 3 x n as expected by pointnet
    input = input.unsqueeze(0).transpose(2, 1)

    # Baseline is zeros, however could also be some kind of noise cloud
    baseline = torch.zeros_like(input)

    attributions = ig.attribute(input, baseline, target=1, internal_batch_size=1)
    
    # Transpose back to n x 3 and remove batch dim
    attributions = attributions.transpose(1, 2).squeeze(0)
    
    # Move to CPU for processing
    attributions = attributions.cpu().numpy()
    
    print(attributions)

# %%
# Sum x, y and z values for an overall attribution for that point
xyz_sum = np.sum(attributions, axis=1)

xyz_sum = np.sign(xyz_sum) * np.power(np.abs(xyz_sum), 0.1)

# Normalise into range -1, 1 such that positive and negative attributions are preserved

norm = Normalize(vmin = -np.max(np.abs(xyz_sum)), vmax = np.max(np.abs(xyz_sum)))

norm_attributions = norm(xyz_sum)

# Cmap for pyvista
cmap = plt.get_cmap('seismic')
colours = cmap(norm_attributions)
colours_rgb = (colours[:, :3] * 255).astype(np.uint8)

pv_cloud = pv.PolyData(cloud)

pv_cloud['colors'] = colours_rgb
    
plotter = pv.Plotter()

plotter.add_points(pv_cloud, scalars='colors', rgb=True)

plotter.set_background("black")

plotter.show()

# %%



