Accompanying code for "Interpretable early Alzheimer’s detection using deep learning on left hippocampus point cloud representations"
==================

The codebase explores deep learning methods on 3d MRI scans for the purpose of classifying between ADNI subjects that are denoted CN (cognitively normal) and MCI (mild cognitive impairment), which is an early stage of Alzheimer's disease. The research conducted using this code was presented at MIUA 2025 and published in Frontiers: 

`https://doi.org/10.3389/978-2-8325-5137-0`

It provides modules for processing NifTi format 3D ADNI (or other MRI) data, modules for easily training pytorch deep learning models on these data, and a skeleton CDSS pipeline used to demonstrate the results of this project.

There are many temporary files in the dev branch which contain the results used to plot the figures in the report. Some of these work with the interactive notebooks. A directory of all figures is also contained in the dev branch.

Please see the API documentation for details on the modules, information on the pipeline usage and a 3D visualisation of left hippocampal attributions:

`https://oscarlloydjohn.github.io/lloyd-john-MIUA-2025/`
