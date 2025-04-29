======================
ozzy_torch_utils
======================

Overview
--------

This package contains a lightweight framework for repeatable and reusable development of PyTorch models. Effectively, it lets you write a run_prediction function and initialise a ModelParameters instance. From here you can easily train models and generate metrics. The models can be saved too.

Modules:
---------------------

.. toctree::
   :maxdepth: 4

   ozzy_torch_utils.init_dataloaders
   ozzy_torch_utils.model_parameters
   ozzy_torch_utils.plot
   ozzy_torch_utils.split_dataset
   ozzy_torch_utils.subject_dataset
   ozzy_torch_utils.train_nn
