#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Input:
#  input NIfTI filename, e.g. DWI.nii
#  full-width half maximum, e.g. 4
#Output
#  Binary NIfTI image with 'z' prefix
#Example Usage
# python dog.py ./DWI.nii 4

import nibabel as nib
from scipy import ndimage
import numpy as np
import scipy.stats as st
import os
import sys
import math
#skimage package is "scikit-image"
import skimage

def bound(lo, hi, val):
    return max(lo, min(hi, val))

def dehaze(img, level):
    """use Otsu to threshold https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_multiotsu.html
        n.b. threshold used to mask image: dark values are zeroed, but result is NOT binary
        level: value 1..5 with larger values preserving more bright voxels
        level: dark_classes/total_classes
            1: 3/4
            2: 2/3
            3: 1/2
            4: 1/3
            5: 1/4
    """
    level = bound(1, 5, level)
    n_classes = abs(3 - level) + 2
    dark_classes = 4 - level
    dark_classes = bound(1, 3, dark_classes)
    thresholds = skimage.filters.threshold_multiotsu(img, n_classes)
    thresh = thresholds[dark_classes - 1]
    print("Zeroing voxels darker than ", thresh)
    img[img < thresh] = 0
    return img

# https://github.com/nilearn/nilearn/blob/1607b52458c28953a87bbe6f42448b7b4e30a72f/nilearn/image/image.py#L164
def _smooth_array(arr, affine, fwhm=None, ensure_finite=True, copy=True):
    """Smooth images by applying a Gaussian filter.

    Apply a Gaussian filter along the three first dimensions of `arr`.

    Parameters
    ----------
    arr : :class:`numpy.ndarray`
        4D array, with image number as last dimension. 3D arrays are also
        accepted.

    affine : :class:`numpy.ndarray`
        (4, 4) matrix, giving affine transformation for image. (3, 3) matrices
        are also accepted (only these coefficients are used).
        If `fwhm='fast'`, the affine is not used and can be None.

    fwhm : scalar, :class:`numpy.ndarray`/:obj:`tuple`/:obj:`list`, 'fast' or None, optional
        Smoothing strength, as a full-width at half maximum, in millimeters.
        If a nonzero scalar is given, width is identical in all 3 directions.
        A :class:`numpy.ndarray`, :obj:`tuple`, or :obj:`list` must have 3 elements,
        giving the FWHM along each axis.
        If any of the elements is zero or None, smoothing is not performed
        along that axis.
        If  `fwhm='fast'`, a fast smoothing will be performed with a filter
        [0.2, 1, 0.2] in each direction and a normalisation
        to preserve the local average value.
        If fwhm is None, no filtering is performed (useful when just removal
        of non-finite values is needed).

    ensure_finite : :obj:`bool`, optional
        If True, replace every non-finite values (like NaNs) by zero before
        filtering. Default=True.

    copy : :obj:`bool`, optional
        If True, input array is not modified. True by default: the filtering
        is not performed in-place. Default=True.

    Returns
    -------
    :class:`numpy.ndarray`
        Filtered `arr`.

    Notes
    -----
    This function is most efficient with arr in C order.

    """
    # Here, we have to investigate use cases of fwhm. Particularly, if fwhm=0.
    # See issue #1537
    if isinstance(fwhm, (int, float)) and (fwhm == 0.0):
        warnings.warn("The parameter 'fwhm' for smoothing is specified "
                      "as {0}. Setting it to None "
                      "(no smoothing will be performed)"
                      .format(fwhm))
        fwhm = None
    if arr.dtype.kind == 'i':
        if arr.dtype == np.int64:
            arr = arr.astype(np.float64)
        else:
            arr = arr.astype(np.float32)  # We don't need crazy precision.
    if copy:
        arr = arr.copy()
    if ensure_finite:
        # SPM tends to put NaNs in the data outside the brain
        arr[np.logical_not(np.isfinite(arr))] = 0
    if isinstance(fwhm, str) and (fwhm == 'fast'):
        arr = _fast_smooth_array(arr)
    elif fwhm is not None:
        fwhm = np.asarray([fwhm]).ravel()
        fwhm = np.asarray([0. if elem is None else elem for elem in fwhm])
        affine = affine[:3, :3]  # Keep only the scale part.
        fwhm_over_sigma_ratio = np.sqrt(8 * np.log(2))  # FWHM to sigma.
        vox_size = np.sqrt(np.sum(affine ** 2, axis=0))
        #n.b. FSL specifies blur in sigma, SPM in FWHM
        # FWHM = sigma*sqrt(8*ln(2)) = sigma*2.3548.
        #convert fwhm to sd in voxels see https://github.com/0todd0000/spm1d
        fwhmvox = fwhm / vox_size
        sd = fwhmvox / math.sqrt(8 * math.log(2))
        for n, s in enumerate(sd):
            if s > 0.0:
                ndimage.gaussian_filter1d(arr, s, output=arr, axis=n)
    return arr

def binary_zero_crossing(img):
    #binarize: negative voxels are zero
    edge = np.where(img > 0.0, 1, 0)
    edge = ndimage.distance_transform_edt(edge)
    edge[edge > 1] = 0
    edge[edge > 0] = 1
    edge = edge.astype('uint8')
    return edge

def difference_of_gaussian(nii, img, fwhmNarrow):
    #apply Difference of Gaussian filter
    # https://en.wikipedia.org/wiki/Difference_of_Gaussians
    # https://en.wikipedia.org/wiki/Marr–Hildreth_algorithm
    #D. Marr and E. C. Hildreth. Theory of edge detection. Proceedings of the Royal Society, London B, 207:187-217, 1980
    #Choose the narrow kernel width
    #  human cortex about 2.5mm thick
    #arbitrary ratio of wide to narrow kernel
    #  Marr and Hildreth (1980) suggest 1.6
    #  Wilson and Giese (1977) suggest 1.5
    #Large values yield smoother results
    fwhmWide = fwhmNarrow * 1.6
    #optimization: we will use the narrow Gaussian as the input to the wide filter
    fwhmWide = math.sqrt((fwhmWide*fwhmWide) - (fwhmNarrow*fwhmNarrow));
    print('Narrow/Wide FWHM {} / {}'.format(fwhmNarrow, fwhmWide))
    img25 = _smooth_array(img, nii.affine, fwhmNarrow)
    img40 = _smooth_array(img25, nii.affine, fwhmWide)
    img = img25 - img40
    img = binary_zero_crossing(img)
    return img

def process_nifti(fnm, fwhm): 
    hdr = nib.load(fnm)
    img = hdr.get_fdata()
    hdr.header.set_data_dtype(np.float32)
    img = img.astype(np.float32)
    str = f'Input intensity range {np.nanmin(img)}..{np.nanmax(img)}'
    print(str)
    str = f'Image shape {img.shape[0]}x{img.shape[1]}x{img.shape[2]}'
    print(str)
    img = dehaze(img, 5)
    img = difference_of_gaussian(hdr, img, fwhm)
    nii = nib.Nifti1Image(img, hdr.affine, hdr.header)
    # print(nii.header)
    #update header
    nii.header.set_data_dtype(np.uint8)  
    nii.header['intent_code'] = 0
    nii.header['scl_slope'] = 1.0
    nii.header['scl_inter'] = 0.0
    nii.header['cal_max'] = 0.0
    nii.header['cal_min'] = 0.0
    pth, nm = os.path.split(fnm)
    if not pth:
        pth = '.'
    outnm = pth + os.path.sep + 'z' + nm
    nib.save(nii, outnm)    

if __name__ == '__main__':
    """Apply Gaussian smooth to image
    Parameters
    ----------
    fnm : str
        NIfTI image to convert
    """
    if len(sys.argv) < 2:
        print('No filename provided: I do not know which image to convert!')
        sys.exit()
    fnm = sys.argv[1]
    fwhm = int(sys.argv[2])
    process_nifti(fnm, fwhm)

