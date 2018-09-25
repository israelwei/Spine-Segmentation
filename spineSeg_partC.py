import nibabel as nib
import numpy as np
import ex1_partb



def save_nifti_image(img_data, filename):
    """
    A function for saving nifti image

    :param img_data: the data of nifti image
    :param filename: the filename of the image to write to
    """
    img = nib.Nifti1Image(img_data, np.eye(4))
    nib.save(img, filename)

def region_growing(CT_data, seed, rad):
    """
    img: ndarray, ndim=3
        An image volume.
    seed: tuple, len=3
        Region growing starts from this point.
    rad: int
        The image neighborhood radius for deciding if including.
    """

    seg = np.zeros(CT_data.shape)
    #seg contains the segmentaion of current seeded region growing
    checked = np.zeros_like(seg)

    seg[seed] = True
    checked[seed] = True
    needs_check = get_neighbors(seed, checked, CT_data.shape)

    while len(needs_check) > 0:
        cur_point = needs_check.pop()

        # It's possible that the point was already checked and was
        # put in the needs_check stack multiple times.
        if checked[cur_point]:
            continue

        checked[cur_point] = True

        # Handle borders.
        imin = max(cur_point[0] - rad, 0)
        imax = min(cur_point[0] + rad, CT_data.shape[0] - 1)
        jmin = max(cur_point[1] - rad, 0)
        jmax = min(cur_point[1] + rad, CT_data.shape[1] - 1)
        kmin = max(cur_point[2] - rad, 0)
        kmax = min(cur_point[2] + rad, CT_data.shape[2] - 1)

        #I chose the homogenity function as the simple mean of neighbors
        if CT_data[cur_point] >= CT_data[imin:imax + 1, jmin:jmax + 1, kmin:kmax + 1].mean():
            # Include the voxel in the segmentation and
            # add its neighbors to be checked.
            seg[cur_point] = True
            needs_check += get_neighbors(cur_point, checked, CT_data.shape)

    return seg.astype(int)


def get_neighbors(cur_vox, checked, dims):
    """
    This function adds neighboring voxels when necessary
    :param cur_vox: current voxel
    :param checked: indicator array of checked points (so we won't add them)
    :param dims: dimensions of the volume
    :return:
    """

    #note: UGLIEST CODE I'VE EVER WRITTEN, SORRY.

    neighbors = []

    #below are all checks for 6-connected voxels, in the first "if"
    # adding 18-connected conditions in the "second-inner if" in each block
    # adding 26-connected conditions in the "third-inner if" in each block

    #x - 1
    if (cur_vox[0] > 0) and not checked[cur_vox[0]-1, cur_vox[1], cur_vox[2]]:
        neighbors.append((cur_vox[0] - 1, cur_vox[1], cur_vox[2]))
        if (cur_vox[1] > 0) and not checked[cur_vox[0] - 1, cur_vox[1] - 1, cur_vox[2]]:
            neighbors.append((cur_vox[0] - 1, cur_vox[1] - 1, cur_vox[2]))
            # x-1, y-1, z-1
            if (cur_vox[2] > 0) and not checked[cur_vox[0] - 1, cur_vox[1] - 1, cur_vox[2] - 1]:
                neighbors.append((cur_vox[0] - 1, cur_vox[1] - 1, cur_vox[2] - 1))
            # x-1, y-1, z+1
            if (cur_vox[2] < dims[2] - 1) and not checked[cur_vox[0] - 1, cur_vox[1] - 1, cur_vox[2] + 1]:
                neighbors.append((cur_vox[0] - 1, cur_vox[1] - 1, cur_vox[2] + 1))
        #x-1, y+1
        if (cur_vox[1] < dims[1] - 1) and not checked[cur_vox[0] - 1, cur_vox[1] + 1, cur_vox[2]]:
            neighbors.append((cur_vox[0] - 1, cur_vox[1] + 1, cur_vox[2]))
            #x-1, y+1, z-1:
            if (cur_vox[2] > 0) and not checked[cur_vox[0] - 1, cur_vox[1] + 1, cur_vox[2] - 1]:
                neighbors.append((cur_vox[0] - 1, cur_vox[1] + 1, cur_vox[2] - 1))
        if (cur_vox[2] > 0) and not checked[cur_vox[0] - 1, cur_vox[1], cur_vox[2] - 1]:
            neighbors.append((cur_vox[0] - 1, cur_vox[1], cur_vox[2] - 1))
        if (cur_vox[2] < dims[2] - 1) and not checked[cur_vox[0] - 1, cur_vox[1], cur_vox[2] + 1]:
            neighbors.append((cur_vox[0] - 1, cur_vox[1], cur_vox[2] + 1))

    #y - 1
    if (cur_vox[1] > 0) and not checked[cur_vox[0], cur_vox[1]-1, cur_vox[2]]:
        neighbors.append((cur_vox[0], cur_vox[1] - 1, cur_vox[2]))
        #y-1, z-1:
        if (cur_vox[2] > 0) and not checked[cur_vox[0], cur_vox[1] - 1, cur_vox[2] - 1]:
            neighbors.append((cur_vox[0], cur_vox[1] - 1, cur_vox[2] - 1))
            #y-1, z-1, x+1:
            if (cur_vox[0] < dims[0] - 1) and not checked[cur_vox[0] + 1, cur_vox[1] - 1, cur_vox[2] - 1]:
                neighbors.append((cur_vox[0] + 1, cur_vox[1] - 1, cur_vox[2] - 1))
        #y-1, z+1:
        if (cur_vox[2] < dims[2] - 1) and not checked[cur_vox[0], cur_vox[1] - 1, cur_vox[2] + 1]:
            neighbors.append((cur_vox[0], cur_vox[1] - 1, cur_vox[2] + 1))


    #z - 1
    if (cur_vox[2] > 0) and not checked[cur_vox[0], cur_vox[1], cur_vox[2]-1]:
        neighbors.append((cur_vox[0], cur_vox[1], cur_vox[2] - 1))



    #x + 1
    if (cur_vox[0] < dims[0]-1) and not checked[cur_vox[0]+1, cur_vox[1], cur_vox[2]]:
        neighbors.append((cur_vox[0] + 1, cur_vox[1], cur_vox[2]))
        if (cur_vox[1] < dims[1]-1) and not checked[cur_vox[0]+1, cur_vox[1]+1, cur_vox[2]]:
            neighbors.append((cur_vox[0] + 1, cur_vox[1] + 1, cur_vox[2]))
            # x+1, y+1, z+1:
            if (cur_vox[2] < dims[2] - 1) and not checked[cur_vox[0] + 1, cur_vox[1] + 1, cur_vox[2] + 1]:
                neighbors.append((cur_vox[0] + 1, cur_vox[1] + 1, cur_vox[2] + 1))
            #x+1, y+1, z-1:
            if (cur_vox[2] > 0) and not checked[cur_vox[0] + 1, cur_vox[1] + 1, cur_vox[2] - 1]:
                neighbors.append((cur_vox[0] + 1, cur_vox[1] + 1, cur_vox[2] - 1))
        #x+1, y-1:
        if (cur_vox[1] > 0) and not checked[cur_vox[0] + 1, cur_vox[1] - 1, cur_vox[2]]:
            neighbors.append((cur_vox[0] + 1, cur_vox[1] - 1, cur_vox[2]))
            # x+1, y-1, z+1:
            if (cur_vox[2] < dims[2] - 1) and not checked[cur_vox[0] + 1, cur_vox[1] - 1, cur_vox[2] + 1]:
                neighbors.append((cur_vox[0] + 1, cur_vox[1] - 1, cur_vox[2] + 1))
        if (cur_vox[2] < dims[2] - 1) and not checked[cur_vox[0] + 1, cur_vox[1], cur_vox[2] + 1]:
            neighbors.append((cur_vox[0] + 1, cur_vox[1], cur_vox[2] + 1))
        if (cur_vox[2] > 0) and not checked[cur_vox[0] + 1, cur_vox[1], cur_vox[2] - 1]:
            neighbors.append((cur_vox[0] + 1, cur_vox[1], cur_vox[2] - 1))
    #y + 1
    if (cur_vox[1] < dims[1]-1) and not checked[cur_vox[0], cur_vox[1]+1, cur_vox[2]]:
        neighbors.append((cur_vox[0], cur_vox[1] + 1, cur_vox[2]))
        #y+1, z+1:
        if (cur_vox[2] < dims[2]-1) and not checked[cur_vox[0], cur_vox[1] + 1, cur_vox[2] + 1]:
            neighbors.append((cur_vox[0], cur_vox[1] + 1, cur_vox[2] + 1))
            #y+1, z+1, x-1:
            if (cur_vox[0] > 0) and not checked[cur_vox[0] - 1, cur_vox[1] + 1, cur_vox[2] + 1]:
                neighbors.append((cur_vox[0] - 1, cur_vox[1] + 1, cur_vox[2] + 1))

        #y+1, z-1:
        if (cur_vox[2] > 0) and not checked[cur_vox[0], cur_vox[1] + 1, cur_vox[2] - 1]:
            neighbors.append((cur_vox[0], cur_vox[1] + 1, cur_vox[2] - 1))

    #z + 1
    if (cur_vox[2] < dims[2]-1) and not checked[cur_vox[0], cur_vox[1], cur_vox[2]+1]:
        neighbors.append((cur_vox[0], cur_vox[1], cur_vox[2] + 1))

    return neighbors


def bones_by_threshold(CT_scan):
    """
    Gets the bones of CT scan from thresholding the image
     between wanted values
    :param CT_scan: image of ct scan
    :return: segmentation of bones in image
    """

    Imin = 250
    Imax = 2000
    bones_seg = CT_scan.get_data()
    bones_seg[bones_seg < Imin] = 0
    bones_seg[bones_seg > Imax] = 0
    bones_seg[bones_seg != 0] = 1

    #saving initial segmentation for sanity check: unmark if saving is wanted
    # input_filename = CT_scan.get_filename()
    # filename_till_point = input_filename.split(".")[0]
    # save_nifti_image(bones_seg, filename_till_point + 'init_bones.nii.gz')

    return bones_seg


def multipleSeedsRG(CT, ROI):
    """
    Creates bones segmentation of CT image using given ROI and multiple seeded region growing.
    :param CT: CT image of format nifti
    :param ROI: ROI image of format nifti
    :return: bones_seg: the bones segmentation
    """

    CT_data = CT.get_data()
    ROI_data = ROI.get_data()
    #label seed points:
    initial_bones_seg = bones_by_threshold(CT)
    bones_seg_intersect_ROI = ROI_data + initial_bones_seg
    bones_seg_intersect_ROI[bones_seg_intersect_ROI != 2] = 0
    bones_seg_intersect_ROI[bones_seg_intersect_ROI == 2] = 1
    some_bones_indices = np.argwhere(bones_seg_intersect_ROI)



    num_bones_indices = some_bones_indices.shape[0]
    rand_indices_in_ROI = np.random.randint(0, num_bones_indices, 200)
    rand_points_in_ROI = some_bones_indices[rand_indices_in_ROI]

    bones_seg = np.zeros(CT_data.shape)
    #growing each seed iteratively:
    for region_num in range(200):
        # print("in seed num: ", region_num)
        cur_point_x = rand_points_in_ROI[region_num][0]
        cur_point_y = rand_points_in_ROI[region_num][1]
        cur_point_z = rand_points_in_ROI[region_num][2]
        cur_point = (cur_point_x, cur_point_y, cur_point_z)
        #sending to region_growing with radius of 1 - closest pixels
        cur_seed_region_growing = region_growing(CT_data, cur_point, 1)
        bones_seg = bones_seg + cur_seed_region_growing
        bones_seg[bones_seg == 2] = 1

    return bones_seg




def segmentBones(ctFileName, AortaFileName, outputFileName):
    """
    This function creates the segmentation of bones from CT image automatically, using function
    from previous ex1 part b.
    :param ctFileName: filename of CT image, nifti file
    :param AortaFileName: filename of aorta image, nifti file
    :param outputFileName: filename of output file of bones segmentation
    :return: segmentation file of bones in the dimensions of given CT
    """

    ROI = ex1_partb.create_ROI_img(ctFileName, AortaFileName)
    CT = nib.load(ctFileName)
    bones_seg = multipleSeedsRG(CT, ROI)
    save_nifti_image(bones_seg, outputFileName)
    output_seg_file = nib.load(outputFileName)
    return output_seg_file



def bounding_box_3d(seg_img):
    """
    Finds the bounding box of segmentation image
    :param seg_img:  binary segmentation image
    :return: indices of 3d bounding box
    """

    ones_indices = np.where(seg_img)
    xmin = np.amin(ones_indices[0])
    xmax = np.amax(ones_indices[0])
    ymin = np.amin(ones_indices[1])
    ymax = np.amax(ones_indices[1])
    zmin = np.amin(ones_indices[2])
    zmax = np.amax(ones_indices[2])

    return (xmin, xmax, ymin, ymax, zmin, zmax)


def evalutateSegmentation(ground_truth_seg, bones_seg_data):
    """
    This function calculates a value that gives us evaluation for our segmentation,
    using a ground truth segmentation of the corresponding L1 vertebra.
    :param ground_truth_seg: ground truth segmentation of L1 vertebra
    :param bones_seg: bones segmentation file from segmentBones
    :return: vol_overlap_diff: volume overlap difference between the two segmentations on the L1
            vertebra
    """

    #creating the bounding box of L1 vertebra:
    # ground_truth_seg_img = nib.load(ground_truth_seg)
    ground_truth_seg_data = ground_truth_seg.get_data()
    (xmin, xmax, ymin, ymax, zmin, zmax) = bounding_box_3d(ground_truth_seg_data)
    #creating mask for the bones seg in the bounding box
    mask = np.zeros_like(ground_truth_seg_data)
    mask[xmin:xmax, ymin:ymax, zmin:zmax] = 1
    bones_seg_L1 = np.multiply(bones_seg_data, mask)
    #calculating volume overlap difference:
    intersection = np.logical_and(bones_seg_L1, ground_truth_seg_data).astype(int)
    num_intersect = np.count_nonzero(intersection)
    num_union = np.count_nonzero(bones_seg_L1) + np.count_nonzero(ground_truth_seg_data)

    vol_overlap_diff = 2*(num_intersect / num_union)
    # error = 1 - vol_overlap_diff (not asked here)
    return vol_overlap_diff




#unmark lines below for running on our input images: (I saved in ex1 partb the ROI in format of "Case{num}_CTmergedROI.nii.gz"
# CT_names = ["Case1_CT.nii.gz" , "Case2_CT.nii.gz","Case3_CT.nii.gz","Case4_CT.nii.gz", "HardCase5_CT.nii.gz","Case5_CT.nii.gz",
#              "HardCase1_CT.nii.gz", "HardCase2_CT.nii.gz", "HardCase3_CT.nii.gz", "HardCase4_CT.nii.gz"]
# ROI_names = ["Case1_CTmergedROI.nii.gz", "Case2_CTmergedROI.nii.gz", "Case3_CTmergedROI.nii.gz", "Case4_CTmergedROI.nii.gz", "HardCase5_CTmergedROI.nii.gz", "Case5_CTmergedROI.nii.gz",
#              "HardCase1_CTmergedROI.nii.gz", "HardCase2_CTmergedROI.nii.gz", "HardCase3_CTmergedROI.nii.gz", "HardCase4_CTmergedROI.nii.gz"]
# for i in range(len(CT_names)):
#     CT = nib.load(CT_names[i])
#     ROI = nib.load(ROI_names[i])
#     seg = multipleSeedsRG(CT, ROI)
#     input_filename = CT.get_filename()
#     filename_till_point = input_filename.split(".")[0]
#     print("in file: ", input_filename)
#     print("saving")
#     save_nifti_image(seg, filename_till_point + '_200-region_growing.nii.gz')
#     if i <= 4:
#         filename_till_CT = input_filename.split("CT")[0]
#         ground_truth = nib.load(filename_till_CT + "L1.nii.gz")
#         vol_overlap_diff = evalutateSegmentation(ground_truth, seg)
#         print("vol overlap diff: ", vol_overlap_diff)
#         print("in file: ", input_filename)


#checks of each volume overlap
# cur_seg = nib.load("Case4_CT_200-region_growing.nii.gz")
# L1_seg = nib.load("Case4_L1.nii.gz")
# vol_overlap_diff = evalutateSegmentation(L1_seg, cur_seg.get_data() )
# print("vol overlap: ", vol_overlap_diff)





