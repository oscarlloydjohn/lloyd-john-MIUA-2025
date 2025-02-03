import fsl.data.dicom
from multiprocessing import Pool


print(fsl.data.dicom.MIN_DCM2NIIX_VERSION)

def main():

    pass

# Loads a dicom MRI and converts it to nifti
def dcm_to_nii():

    #fsl.data.dicom
    #NB this is just a wrapper around dcm2niix

    pass

def mri_to_tensor():

    pass

def brain_extraction():

    pass

# Align all MRIs to the same anatomical space
def affine_registration():

    #fsl.transform.affine

    pass

# Remove excess space around the image
def crop():

    pass


def intensity_normalisation():

    pass

