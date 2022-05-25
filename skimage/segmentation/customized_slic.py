import collections as coll
import numpy as np
from scipy import ndimage as ndi
import cv2

from ..util import img_as_float, regular_grid
from ..segmentation._slic import (_slic_cython,
                                  _enforce_label_connectivity_cython)
from ..color import rgb2lab


def draw_centers(img, segments, color):
    img = img.copy()
    for c in segments:
        center_coordinates = (int(c[2]), int(c[1]))
        cv2.circle(img, center_coordinates, 1, color, 2)
    return img


def weighted_average(segments, interest_point, alpha=0.1, m=2):
    """ move centers towards interest point. 
    movement is wrt alpha and the (1/ distance between
    segment[i] and interest_point)**m
    """
    
    x, y = interest_point
    point = np.array([0, y, x, 0,0,0], dtype=np.float64)
    
    dst = np.array([[0]], dtype=np.float64)
    for c in segments:
        temp = np.array([(np.sum((c-point)**2))**0.5])
        dst = np.vstack((dst, temp))
    dst = dst[1:, :]
    dst = dst/np.max(dst)
    dst = 1-dst # the points which are near interest_point 
                # will move more toeards it
    
    segments = alpha*point*dst**m + (1-alpha*dst**m)*segments
    
    return segments


def slic_customized(image, n_segments=100, compactness=10., max_iter=10,
         sigma=0, spacing=None, multichannel=True, convert2lab=None,
         enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3,
         slic_zero=False):
    """Segments image using k-means clustering in Color-(x,y,z) space.
    """

    org_img = image.copy()
    image = img_as_float(image)
    is_2d = False
    if image.ndim == 2:
        # 2D grayscale image
        image = image[np.newaxis, ..., np.newaxis]
        is_2d = True
    elif image.ndim == 3 and multichannel:
        # Make 2D multichannel image 3D with depth = 1
        image = image[np.newaxis, ...]
        is_2d = True
    elif image.ndim == 3 and not multichannel:
        # Add channel as single last dimension
        image = image[..., np.newaxis]

    if spacing is None:
        spacing = np.ones(3)
    elif isinstance(spacing, (list, tuple)):
        spacing = np.array(spacing, dtype=np.double)

    if not isinstance(sigma, coll.Iterable):
        sigma = np.array([sigma, sigma, sigma], dtype=np.double)
        sigma /= spacing.astype(np.double)
    elif isinstance(sigma, (list, tuple)):
        sigma = np.array(sigma, dtype=np.double)
    if (sigma > 0).any():
        # add zero smoothing for multichannel dimension
        sigma = list(sigma) + [0]
        image = ndi.gaussian_filter(image, sigma)

    if multichannel and (convert2lab or convert2lab is None):
        if image.shape[-1] != 3 and convert2lab:
            raise ValueError("Lab colorspace conversion requires a RGB image.")
        elif image.shape[-1] == 3:
            image = rgb2lab(image)

    depth, height, width = image.shape[:3]

    # initialize cluster centroids for desired number of segments
    grid_z, grid_y, grid_x = np.mgrid[:depth, :height, :width]
    slices = regular_grid(image.shape[:3], n_segments)
    step_z, step_y, step_x = [int(s.step if s.step is not None else 1)
                              for s in slices]
    segments_z = grid_z[slices]
    segments_y = grid_y[slices]
    segments_x = grid_x[slices]

    segments_color = np.zeros(segments_z.shape + (image.shape[3],))
    segments = np.concatenate([segments_z[..., np.newaxis],
                               segments_y[..., np.newaxis],
                               segments_x[..., np.newaxis],
                               segments_color],
                              axis=-1).reshape(-1, 3 + image.shape[3])
    segments = np.ascontiguousarray(segments)

    # Draw initial segments center
    img_center = draw_centers(org_img, segments, (100, 250, 50))
    cv2.imshow("org center", img_center)
    
    # change segments by using weighted average wrt its center
    segments = weighted_average(segments, (image.shape[2]/2, image.shape[1]/2), 0.5, 2)
    img_center = draw_centers(img_center, segments, (100, 50, 250))
    cv2.imshow("my center", img_center)
    cv2.waitKey(0)

    # we do the scaling of ratio in the same way as in the SLIC paper
    # so the values have the same meaning
    step = float(max((step_z, step_y, step_x)))
    ratio = 1.0 / compactness

    image = np.ascontiguousarray(image * ratio)

    labels = _slic_cython(image, segments, step, max_iter, spacing, slic_zero)
    
    if enforce_connectivity:
        segment_size = depth * height * width / n_segments
        min_size = int(min_size_factor * segment_size)
        max_size = int(max_size_factor * segment_size)
        labels = _enforce_label_connectivity_cython(labels,
                                                    min_size,
                                                    max_size)

    if is_2d:
        labels = labels[0]
    
    return labels
