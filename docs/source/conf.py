import os
import sys
from sphinx.ext import apidoc

# Add the high-level directory (where your Python modules are) to sys.path
sys.path.insert(0, os.path.abspath("../.."))

# List of modules to document
MODULES_TO_DOCUMENT = [
    "preprocessing_pre_fastsurfer",
    "preprocessing_post_fastsurfer",
    "ozzy_torch_utils",
    "pipeline_utils",
    "final_models_explainability",
]

def run_apidoc(_):
    """Automatically generate .rst files from Python modules."""
    from sphinx.ext.apidoc import main

    docs_source_dir = os.path.abspath(os.path.dirname(__file__))  # /docs/source
    modules_dir = os.path.abspath(os.path.join(docs_source_dir, "../.."))  # High-level code directory

    # Generate .rst for each module inside a subdirectory
    for module in MODULES_TO_DOCUMENT:
        module_output_path = os.path.join(docs_source_dir, module)  # /docs/source/module_name
        module_base_path = os.path.join(modules_dir, module)  # ../module_name

        # Ensure output directory exists
        os.makedirs(module_output_path, exist_ok=True)

        # Run sphinx-apidoc
        apidoc_args = [
            "--force",  # Overwrite existing .rst files
            #"--module-first",  # Place module docstring before submodules
            "--separate",  # Create separate .rst files for each module
            "-o", module_output_path,  # Output directory for .rst files
            module_base_path,  # Path to the Python package/module
        ]

        main(apidoc_args)

def setup(app):
    """Hook to auto-run sphinx-apidoc before building docs."""
    app.connect("builder-inited", run_apidoc)

# -- Project information -----------------------------------------------------
project = 'lloyd-john-MIUA-2025'
copyright = '2025, Oscar Lloyd-John'
author = 'Oscar Lloyd-John'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
]

add_module_names = False

# autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']
