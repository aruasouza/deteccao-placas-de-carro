import numpy as np
import cv2
from skimage.measure import label, regionprops_table

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# def trashold(img, percentile, reverse=False):
#     height, width = img.shape
#     part_width = width // 7
#     result = np.zeros_like(img)
#     for i in range(7):
#         start = i * part_width
#         end = width if i == 6 else (i + 1) * part_width
#         part = img[:, start:end]
#         if reverse:
#             th = np.percentile(part, 100 - percentile)
#             _, trasholded = cv2.threshold(part, th, 1.0, cv2.THRESH_BINARY)
#         else:
#             th = np.percentile(part, percentile)
#             _, trasholded = cv2.threshold(part, th, 1.0, cv2.THRESH_BINARY_INV)
#         result[:, start:end] = trasholded
#     return result


def split_chars(img,nchars,margin=40):
    m2 = margin // 2
    height,width = img.shape
    estimated_cut_points = [i*(width//nchars) for i in range(1,nchars)]
    cut_points = [0]
    for point in estimated_cut_points:
        start,end = point - m2,point + m2
        white = 0
        better_point = point
        for p in range(start,end):
            total_white = np.sum(img[:,p] == 0.0)
            if total_white > white:
                white = total_white
                better_point = p
        cut_points.append(better_point)
    cut_points.append(width)
    final_images = []
    for c0,c1 in zip(cut_points[:-1], cut_points[1:]):
        let = img[:,c0:c1]
        h,w = let.shape
        sobra = (height - w) // 2
        canvas = np.zeros((height,height))
        canvas[:, sobra:sobra + w] = img[:,c0:c1]
        final_images.append(canvas)
    return final_images

def split_chars_2(img,nchars,margin=5):
    height,width = img.shape
    estimated_cut_points = [i*(width//nchars) for i in range(0,nchars+1)]
    images = [img[:,max(c1-margin,0):c2+margin] for c1,c2 in zip(estimated_cut_points[:-1], estimated_cut_points[1:])]
    canvas = np.ones((height, 2*margin+(width//nchars))) * img.mean()
    canv1 = canvas.copy()
    canv1[:,:images[0].shape[1]] = images[0]
    images[0] = canv1
    canv2 = canvas.copy()
    canv2[:,:images[-1].shape[1]] = images[-1]
    images[-1] = canv2
    return images

def clean_objects(img):
    labeled = label(img)
    props = regionprops_table(labeled, properties=['area'])
    areas = props['area']
    if len(areas) == 0:
        return img
    max_area_label = np.argmax(areas) + 1
    cleaned = img.copy()
    cleaned[labeled != max_area_label] = 0.0
    return cleaned

def sauvola_threshold(img, window_size=201, k=0.1, R=0.5):
    if window_size % 2 == 0:
        window_size += 1
    img_f = img.astype(np.float64)
    mean = cv2.boxFilter(img_f, ddepth=-1, ksize=(window_size, window_size),
                         normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(img_f ** 2, ddepth=-1,
                            ksize=(window_size, window_size),
                            normalize=True, borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
    threshold = mean * (1.0 + k * (std / R - 1.0))
    binarized = np.where(img_f < threshold, 1.0, 0.0)
    return binarized.astype(np.float32)

def trashold(img, percentile, reverse=False, window_size=201, k=0.005, R=0.5):
    if reverse:
        binarized = sauvola_threshold(1.0 - img, window_size, k, R)
    else:
        binarized = sauvola_threshold(img, window_size, k, R)
    return binarized