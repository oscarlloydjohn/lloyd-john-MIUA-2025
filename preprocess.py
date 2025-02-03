import fsl.data.dicom
import fsl.data.image
import fsl.wrappers.bet
import fsl.transform.affine
from multiprocessing import Pool

def main():

    pass

# Loads a dicom MRI and converts it to nifti
def dcm_to_nii():

    #fsl.data.dicom
    #NB this is just a wrapper around dcm2niix

    print(fsl.data.dicom.dcm2niix())
    print(fsl.data.dicom.enabled())

    metadata = fsl.data.dicom.scanDir("/uolstore/home/student_lnxhome01/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/sample-mri/I10968663")

    dicomImage = fsl.data.dicom.loadSeries(metadata[0])

    print(type(dicomImage[0]))

    #print(myimage.__str__)

    #fsl.wrappers.bet("nii-bet-test/I1282317/I1282317_Sagittal_3D_FLAIR_20191016160427_3.nii", "nii-bet-test/I1282317/wrapper-out.nii")

    #fsl.data.dicom.__init__(myimage, ,"sample-mri/I10968663")

    #myimage.save

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

dcm_to_nii()

