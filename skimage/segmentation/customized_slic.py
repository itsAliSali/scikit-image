import collections as coll
import numpy as np
from scipy import ndimage as ndi
import cv2

from ..util import img_as_float, regular_grid
from ..segmentation._slic import (_slic_cython,
                                  _enforce_label_connectivity_cython)
from ..color import rgb2lab
from .boundaries import mark_boundaries

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

segments =np.array([])
mode = 'nothing'
perform_slic = True

def click_and_crop(event, x, y, flags, param):
    global segments, mode, perform_slic 

    point = np.array([0, y, x, 0,0,0], dtype=np.float64)

    if event == cv2.EVENT_LBUTTONDOWN:
        approximity_tresh = 15
        err = np.sum((segments-point)**2, axis=1, keepdims=True)
        mask = err < approximity_tresh
        mask = np.repeat(mask, 6, axis=1)
        touched_a_point = np.sum(mask) != 0
        if touched_a_point:
            # clicked_point = segments[mask]
            clicked_ind = np.where(mask == True)[0][0]
            # print(clicked_point, clicked_ind)
            # print('seg', segments.shape)
            if mode == 'delete' or mode == 'move':
                segments = np.vstack((segments[:clicked_ind], segments[clicked_ind+1:]))
                # print('new seg1', segments.shape)
                perform_slic = True

        if mode == 'add':
            segments = np.vstack((segments, point))
            # print('new seg2', segments.shape)
            perform_slic = True

    elif event == cv2.EVENT_LBUTTONUP:
        if mode == 'move':
            segments = np.vstack((segments, point))
            # print('new seg3', segments.shape)
            perform_slic = True
        elif mode == 'focus':
            segments = weighted_average(segments, (x, y), 0.3, 4)
            perform_slic = True
            

def slic_customized(image, n_segments=100, compactness=10., max_iter=10,
         sigma=0, spacing=None, multichannel=True, convert2lab=None,
         enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3,
         slic_zero=False):
    """Segments image using k-means clustering in Color-(x,y,z) space.
    """
    global segments, mode
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
    # img_center = draw_centers(org_img, segments, (100, 250, 50))
    # cv2.imshow("org center", img_center)
    
    # change segments by using weighted average wrt its center
    # segments = weighted_average(segments, (320, 230), 0.3, 2)
    # segments = weighted_average(segments, (320, 230), 0.5, 4)
    # img_center = draw_centers(img_center, segments, (100, 50, 250))
    # cv2.imshow("my center", img_center)
    # cv2.waitKey(0)

    cv2.namedWindow("initial centers")
    cv2.setMouseCallback("initial centers", click_and_crop)
    step = float(max((step_z, step_y, step_x)))
        
    global perform_slic

    while True:
        
        img_center = draw_centers(org_img, segments, (100, 50, 250))
        cv2.imshow("initial centers", img_center)
        
        # we do the scaling of ratio in the same way as in the SLIC paper
        # so the values have the same meaning
        # step = float(max((step_z, step_y, step_x))) *2
        ratio = 1.0 / compactness
        image_r = np.ascontiguousarray(image * ratio)

        if perform_slic:
            perform_slic = False
            labels = _slic_cython(image_r, segments.copy(), step, max_iter+10, spacing.copy(), slic_zero)
            if enforce_connectivity:
                segment_size = depth * height * width / n_segments
                min_size = int(min_size_factor * segment_size)
                max_size = int(max_size_factor * segment_size)
                labels = _enforce_label_connectivity_cython(labels, min_size, max_size)
          
            if is_2d:
                labels = labels[0]
            print('number of patches:', len(np.unique(labels)))
            labels = cv2.medianBlur(np.uint8(labels), 5)
        
        cv2.imshow('labels', np.uint8(labels/np.max(labels)*255))
        B = mark_boundaries(org_img, labels)
        cv2.imshow('bounbaries', B)

        # img_B_center = draw_centers(np.uint8(B*255), segments, (100, 50, 250))
        # cv2.imshow("seg+B", img_B_center)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break
        elif key == ord("x"):
            compactness +=5
            perform_slic = True
            print("compactness: ", compactness)
        elif key == ord("c"):
            if compactness >5:
                perform_slic = True
                compactness -=5
                print("compactness: ", compactness)
        elif key == ord("e"):
            perform_slic = True
            enforce_connectivity = not enforce_connectivity
            print("enforce_connectivity: ", enforce_connectivity)
        elif key == ord("v"):
            perform_slic = True
            min_size_factor += 0.025
            print("min_size_factor: ", min_size_factor)
        elif key == ord("b"):
            if min_size_factor >= 0.025:
                perform_slic = True
                min_size_factor -= 0.025
                print("min_size_factor: ", min_size_factor)
        elif key == ord("d"):
            mode = 'delete'
        elif key == ord("a"):
            mode = 'add'
        elif key == ord("m"):
            mode = 'move'
        elif key == ord("f"):
            mode = 'focus'
        elif key == ord("g"):
            step +=5
            perform_slic = True
            print("step: ", step)
        elif key == ord("h"):
            if step >5:
                perform_slic = True
                step -=5
                print("step: ", step)

    return labels
