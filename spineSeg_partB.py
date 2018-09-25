import nibabel as nib
import numpy as np
import scipy
from scipy import ndimage
import skimage
from skimage import measure
from skimage.morphology import convex_hull


def save_nifti_image(img_data, filename):
    """
    A function for saving nifti image

    :param img_data: the data of nifti image
    :param filename: the filename of the image to write to
    """
    img = nib.Nifti1Image(img_data, np.eye(4))
    nib.save(img, filename)


def IsolateBody(CT_scan):
    """
    Isolates the body from the image
    :param CT_scan: ct scan of image
    :return: binary body segmentation image
    """

    #thresholding
    Imin = -500
    Imax = 2000
    body_seg = CT_scan.get_data()
    body_seg[body_seg < Imin] = 0
    body_seg[body_seg > Imax] = 0
    body_seg[body_seg != 0] = 1

    #removing small area pixels
    # #connected components:
    labeled_components_seg, cur_connected_components = \
        skimage.measure.label(body_seg, return_num=True)
    id_sizes = np.array(ndimage.sum(body_seg, labeled_components_seg, range(cur_connected_components + 1)))
    area_mask_small_areas = (id_sizes < 70)
    body_seg[area_mask_small_areas[labeled_components_seg]] = 0

    #noise filtering
    body_seg = scipy.ndimage.binary_opening(body_seg).astype(int)

    #returning the largest connected component:
    labeled_components_seg, cur_connected_components = \
        skimage.measure.label(body_seg, return_num=True)
    id_sizes = np.array(ndimage.sum(body_seg, labeled_components_seg, range(cur_connected_components + 1)))
    biggest_component = np.amax(id_sizes)
    area_mask = (id_sizes < biggest_component)
    body_seg[area_mask[labeled_components_seg]] = 0
    input_filename = CT_scan.get_filename()
    filename_till_point = input_filename.split(".")[0]
    save_nifti_image(body_seg, filename_till_point + "body_seg.nii.gz")

    return body_seg



def IsolateBS(body_seg):
    """
    Isolates the breathing system from the body
    :param body_seg: biggest connected component of body segmentation
    :return:
    """

    #identifying the holes inside the body segmentation
    inverse_body_seg = 1 - body_seg
    labeled_components_inverse_seg, cur_connected_components_inverse = \
        skimage.measure.label(inverse_body_seg, return_num=True)
    id_sizes = np.array(ndimage.sum(inverse_body_seg, labeled_components_inverse_seg,
                                    range(cur_connected_components_inverse + 1)))
    second_largest_component = np.partition(id_sizes, -2)[-2]
    area_mask = (id_sizes != second_largest_component)
    inverse_body_seg[area_mask[labeled_components_inverse_seg]] = 0
    breathing_seg = inverse_body_seg

    #identifying the slices:
    #inferior slice- BB:
    dim1, dim2, dim3 = breathing_seg.shape

    lowest_z = 0
    flag_z = 0
    widest_slice_size = 0
    widest_slice_z_id = 0
    for z in range(dim3):
        if np.count_nonzero(breathing_seg[:, :, z]) > 0:
            if not flag_z:
                flag_z = 1
                lowest_z = z
                # widest slice- CC:
                widest_slice_z_id = lowest_z
                widest_slice_size = np.count_nonzero(breathing_seg[:, :, lowest_z])
                continue
            pixels_in_slice = np.count_nonzero(breathing_seg[:, :, z])
            if pixels_in_slice > widest_slice_size:
                widest_slice_z_id = z
                widest_slice_size = pixels_in_slice
        else:
            if flag_z:
                break

    slice_BB_index = lowest_z

    #widest slice / upper slice:
    slice_CC_index = widest_slice_z_id

    return breathing_seg, slice_BB_index, slice_CC_index




def ThreeDBand(body_seg, breathing_seg, BB_index, CC_index):
    """
    Creates a 3d band between the convex hull of the breathing system and
    the gap to the body segmentation
    :param body_seg:
    :param breathing_seg:
    :param BB_index:
    :param CC_index:
    :return:
    """

    max_z = CC_index
    breathing_seg[:, :, max_z:] = 0 #clearing the values beyond CC slice
    min_z = BB_index
    breathing_seg_copy = breathing_seg
    # convex hull of breathing system segmantation:
    convex_hull_seg = np.zeros(breathing_seg_copy.shape)

    for z in range(min_z, max_z):
        convex_hull_seg[:, :, z] = convex_hull.convex_hull_image(breathing_seg_copy[:, :, z]).astype(int)
    body_seg[:, :, :min_z] = 0  #clearing the values of body_seg below BB slice
    body_seg[:, :, max_z:] = 0 #clearing the values of body_seg beyond CC slice

    confined_region = body_seg - convex_hugfll_seg
    confined_region[confined_region < 0] = 0

    return confined_region



def spine_by_threshold(CT_scan):
    """
    Gets the bones of CT scan from thresholding the image
     between wanted values
    :param CT_scan: image of ct scan
    :return: segmentation of bones in image
    """

    Imin = 500
    Imax = 2000
    spine_seg = CT_scan.get_data()
    spine_seg[spine_seg < Imin] = 0
    spine_seg[spine_seg > Imax] = 0
    spine_seg[spine_seg != 0] = 1

    return spine_seg



def SpineROI(Aorta_segmentation, CT_scan):
    """
    This function returns the ROI of the spine.
    :param ROI_aorta: segmentation of Aorta
    :param CT_scan: image of CT scan
    :returns ROI_aorta: "folding" 3d-rectangle around the aorta in image, using
            region-growing algorithm
    """

    spine_seg_partial = spine_by_threshold(CT_scan)
    Aorta_seg_data = Aorta_segmentation.get_data()

    wanted_shift = 70
    aorta_shift1 = np.roll(Aorta_seg_data, shift=wanted_shift, axis=1)
    aorta_shift2 = np.roll(Aorta_seg_data, shift=-wanted_shift, axis=1)
    spine_ROI_1 = scipy.ndimage.binary_dilation(aorta_shift1, iterations=8,
                                              structure=np.ones((9, 9, 9))) .astype(aorta_shift1.dtype)
    spine_ROI_2 = scipy.ndimage.binary_dilation(aorta_shift2, iterations=8,
                                              structure=np.ones((9, 9, 9))).astype(aorta_shift2.dtype)
    intersect_bones1 = np.logical_and(spine_ROI_1, spine_seg_partial).astype(int)
    intersect_bones2 = np.logical_and(spine_ROI_2, spine_seg_partial).astype(int)
    number_intersect1 = np.count_nonzero(intersect_bones1)
    number_intersect2 = np.count_nonzero(intersect_bones2)
    if number_intersect1 > number_intersect2:
        spine_ROI = spine_ROI_1
    else:
        spine_ROI = spine_ROI_2

    #
    # remarks for saving image
    # input_filename = Aorta_segmentation.get_filename()
    # filename_till_point = input_filename.split(".")[0]
    # save_nifti_image(spine_ROI, filename_till_point + "_spine_ROI.nii.gz")
    return spine_ROI


def MergedROI(confined_ROI, spine_ROI, CT_scan):
    """
    Build a merged ROI that contains the spine and the chest.
    :param spine_ROI: ROI of spine
    :param confined_body_ROI: ROI between the lungs and the body
    :param CT_scan: image of CT scan
    :return: merged ROI, and saves the output
    """

    merged_ROI_data = np.logical_or(spine_ROI, confined_ROI).astype(int)
    input_filename = CT_scan.get_filename()
    filename_till_point = input_filename.split(".")[0]
    save_nifti_image(merged_ROI_data, filename_till_point + "_ROI.nii.gz")

    return merged_ROI_data





def create_ROI_img(ctFileName, AortaFileName):
    """
    This function runs the code on given CT image and creates the ROI of chest + spine
    :param ctFileName: filename of CT image, nifti file
    :param AortaFileName: filenaem of aorta image, nifti file
    """
    filename_till_point = ctFileName.split(".")[0]
    cur_ct = nib.load(ctFileName)
    body_seg = IsolateBody(cur_ct)
    bs_seg, BB_slice, CC_slice = IsolateBS(body_seg)
    confined_ROI = ThreeDBand(body_seg, bs_seg, BB_slice, CC_slice)
    aorta_img = nib.load(AortaFileName)
    spine_ROI = SpineROI(aorta_img, cur_ct)
    merged_ROI_fin = MergedROI(confined_ROI, spine_ROI, cur_ct)
    return merged_ROI_fin



def run_and_save_reg_case(save_image=False):
    """
    This function runs the code on regular cases given
    I assume nave fromats for Aorta files similar to what's given to us
    :param save_image: deciding if saving some images outputs
    """
    for i in range(1, 6):
        filename_input = "Case" + str(i) +"_CT.nii.gz"
        filename_till_point = filename_input.split(".")[0]
        cur_ct = nib.load(filename_input)
        body_seg = IsolateBody(cur_ct)
        bs_seg, BB_slice, CC_slice = IsolateBS(body_seg)
        confined_ROI = ThreeDBand(body_seg, bs_seg, BB_slice, CC_slice)
        if save_image:
            save_nifti_image(confined_ROI, filename_till_point + "_confined_ROI.nii.gz")
        filename_till_CT = filename_input.split("CT")[0]
        aorta_img = nib.load(filename_till_CT + "Aorta.nii.gz")
        spine_ROI = SpineROI(aorta_img, cur_ct)
        merged_ROI_fin = MergedROI(confined_ROI, spine_ROI, cur_ct)



def run_and_save_hard_case(save_image=False):
    """
    This function runs the code on regular cases given
    I assume nave fromats for Aorta files similar to what's given to us
    :param save_image: deciding if saving some images outputs
    """
    for i in range(1, 6):
        filename_input = "HardCase" + str(i) + "_CT.nii.gz"
        filename_till_point = filename_input.split(".")[0]
        cur_ct = nib.load(filename_input)
        body_seg = IsolateBody(cur_ct)
        bs_seg, BB_slice, CC_slice = IsolateBS(body_seg)
        confined_ROI = ThreeDBand(body_seg, bs_seg, BB_slice, CC_slice)
        if save_image:
            save_nifti_image(confined_ROI, filename_till_point + "_confined_ROI.nii.gz")
        filename_till_CT = filename_input.split("CT")[0]
        aorta_img = nib.load(filename_till_CT + "Aorta.nii.gz")
        spine_ROI = SpineROI(aorta_img, cur_ct)
        merged_ROI_fin = MergedROI(confined_ROI, spine_ROI, cur_ct)


#running on input images: unmark each line below
# run_and_save_reg_case(True)
# run_and_save_hard_case(True)
