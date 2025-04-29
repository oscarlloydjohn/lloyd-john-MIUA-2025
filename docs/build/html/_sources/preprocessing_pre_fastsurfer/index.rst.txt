======================
preprocessing_pre_fastsurfer
======================

Overview
--------

This is not a package, rather it contains standalone scripts to manipulate data (usually from ADNI) and process it in parallel using FastSurfer. It also has a utility for selecting a single image per subject. The code has entry points for command line use, however they are implemented such that they can be imported and used in other scripts.

See the :doc:`run_processing` for how to use bash script that wraps the FastSurfer parallel processing in an appropriate apptainer. The script itself is contained inside preprocessing_pre_fastsurfer in the main repo.

Modules:
---------------------

.. toctree::
   :maxdepth: 4

   preprocessing_pre_fastsurfer.compact_dir
   preprocessing_pre_fastsurfer.create_holdout
   preprocessing_pre_fastsurfer.delete_files_called
   preprocessing_pre_fastsurfer.delete_files_containing
   preprocessing_pre_fastsurfer.isolate_adni_screening
   preprocessing_pre_fastsurfer.preprocess
