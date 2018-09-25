import nibabel as nib
import numpy as np
import scipy
from scipy import ndimage
from scipy import sparse
from scipy import stats
import matplotlib.pyplot as plt
import skimage
from skimage import measure



def save_nifti_image(img_data, filename):
    """
    A function for saving nifti image

    :param img_data: the data of nifti image
    :param filename: the filename of the image to write to
    """
    img = nib.Nifti1Image(img_data, np.eye(4))
    nib.save(img, filename)


def SegmentationByTH(img_file, Imin, Imax):
    """
    Creating binary segmentation image. Values between Imin and Imax will be 1,
    other values will be 0.
    :param filename: string of filename
    :param Imin: low threshold for segmentation
    :param Imax: high threshold for segmentation
    :return: the image file after applying segmentation to it by thresholding
    """

    #getting the 3d matrix of scan
    filename_till_point = img_file.get_filename().split(".")[0]
    img_data = img_file.get_data()
    # threshold :
    img_data[img_data < Imin] = 0
    img_data[img_data > Imax] = 0
    img_data[img_data != 0] = 1

    output_filename = filename_till_point + "_seg_" + str(Imin) + "_" + str(Imax) + ".nii.gz"
    save_nifti_image(img_data, output_filename)
    return img_file

def SkeletonTHFinder(filename, debug_flag=False):
    """
    This function iteratively finds the optimal Imin for current CT image
    :param filename:
    :param debug_flag: parameter used for debugging, printing values on the run such as
            best Imin and final number of connected component. default is False (no printing)
    :return: best_Imin: the Imin value that corresponds to lowest value of number of
                        connected components (before all the cleaning)
    """

    threshold_range = [i for i in range(150, 500, 5)]
    num_connected_components = np.zeros(70)
    skeleton_Imax = 1300

    for i in range(70):
        main_img_file = nib.load(filename)
        cur_segmentation = SegmentationByTH(main_img_file, threshold_range[i], skeleton_Imax)
        cur_seg_data = cur_segmentation.get_data()
        # np.asarray(cur_segmentation)
        #connected components of current segmentation:
        labeled_components_seg, cur_connected_components = \
            skimage.measure.label(cur_seg_data, return_num=True)
        num_connected_components[i] = cur_connected_components


    #plotting the wanted graph
    plt.figure()
    plt.plot(threshold_range, num_connected_components, 'o')
    plt.xlabel('Imin')
    plt.ylabel('Number of connected components')
    plt.title('Number of connected components as a function of Imin')
    plt.grid(True)
    filename_till_point = filename.partition(".")[0]
    graph_filename = filename_till_point + "_Graph.jpg"
    #saving the current figure
    plt.savefig(graph_filename)

    # we want Imin that gives the minimum value of connected components:
    wanted_index = np.argmin(num_connected_components)
    best_Imin = threshold_range[int(wanted_index)]
    if debug_flag:
        #printing the best Imin for debugging and
        print(filename_till_point + " best Imin: ", str(best_Imin))
    # taking the segmentation with best Imin
    best_Imin_seg = nib.load(filename_till_point + "_seg_" + str(best_Imin) + "_1300.nii.gz")
    best_seg_data = best_Imin_seg.get_data()


    #morphological operators for filling holes and connecting components:
    best_seg_data = scipy.ndimage.morphology.binary_fill_holes(best_seg_data).astype(best_seg_data.dtype)
    best_seg_data = scipy.ndimage.morphology.binary_erosion(best_seg_data).astype(best_seg_data.dtype)
    best_seg_data = scipy.ndimage.morphology.binary_dilation(best_seg_data, iterations=8).astype(best_seg_data.dtype)
    best_seg_data = scipy.ndimage.morphology.binary_fill_holes(best_seg_data).astype(best_seg_data.dtype)

    # removing lonely pixels and taking the largest connected component:
    labeled_components_seg, cur_connected_components = \
        skimage.measure.label(best_seg_data, return_num=True)
    # in each component, I sum the values of the original image (summing is done in each
    # region separately). The sum gives as the size of the region - I check if its size is less
    # than the biggest component.
    id_sizes = np.array(ndimage.sum(best_seg_data, labeled_components_seg, range(cur_connected_components + 1)))
    biggest_component = np.amax(id_sizes)
    area_mask = (id_sizes < biggest_component)
    best_seg_data[area_mask[labeled_components_seg]] = 0

    if debug_flag:
        labeled_components_seg, cur_connected_components = \
            skimage.measure.label(best_seg_data, return_num=True)
        print("lowest num conneccted components in " + filename_till_point + " after cleaning: "
              + str(cur_connected_components))

    best_seg_filename = filename_till_point + "_SkeletonSegmentation.nii.gz"
    nib.save(best_Imin_seg, best_seg_filename)
    return best_Imin



#running functions above: all Case{1-5} files
# for running the function with new filename - provide first argument with your filename!
case_numbers = [i for i in range(1, 6)]
for i in range(len(case_numbers)):
    cur_best_Imin = SkeletonTHFinder("Case" + str(case_numbers[i]) + "_CT.nii.gz")


