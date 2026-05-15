import cv2
import numpy as np
import utils.me_pipeline as me_pipeline
import utils.br_pipeline as br_pipeline
from utils.inferencia import ONNXYOLO
import os
import time
from utils.placas_pipeline import inferencia_me,inferencia_br

env = os.getenv('ENV', 'DEV')

if env == 'PROD':
    from utils.inferencia import RKNNYOLO
    model = RKNNYOLO('rknn_models/yolo-pose.rknn')
else:
    model = ONNXYOLO('onnx_models/yolo-pose.onnx')

def full_pipeline(img, model=model):
    start = time.time()
    deteccao = detect_placa(model, img)
    detect_time = time.time() - start
    if deteccao is None:
        return None
    start = time.time()
    gray = grayscale(img)
    croped_img = get_croped_image(deteccao, gray)
    warped_img = warp_image(deteccao, croped_img)
    if warped_img is None:
        return None
    preprocessing_time = time.time() - start
    if deteccao['classe'] == 1:
        final_output = br_pipeline.pipeline(warped_img, 30)
        # final_output = inferencia_br(warped_img)
    else:
        final_output = me_pipeline.pipeline(warped_img, 35)
        # final_output = inferencia_me(warped_img)
    final_output['tempo']['tempo_deteccao'] = detect_time
    final_output['tempo']['tempo_extracao'] += preprocessing_time
    return final_output
    
def extract_characters(img, model=model):
    deteccao = detect_placa(model, img)
    if deteccao is None:
        return None
    try:
        gray = grayscale(img)
        croped_img = get_croped_image(deteccao, gray)
        warped_img = warp_image(deteccao, croped_img)
        if deteccao['classe'] == 1:
            final_output = br_pipeline.extract_pipeline(warped_img, 30)
        else:
            final_output = me_pipeline.extract_pipeline(warped_img, 35)
        return final_output
    except:
        return None

def detect_placa(model, img, minconf = 0):
    outputs, (orig_h, orig_w) = model(img)
    if not outputs or len(outputs) == 0:
        return None
    input_h, input_w = model.input_shape[2:]
    scale_x = orig_w / input_w
    scale_y = orig_h / input_h
    detections = outputs[0][0]
    if len(detections) == 0:
        return None
    best_det = detections[0].T
    x1, y1, x2, y2 = best_det[:4]
    conf = best_det[4]
    if conf < minconf:
        return None
    classe = int(best_det[5])
    kp_data = best_det[6:]
    kp = [(int(kp_data[i] * scale_x), int(kp_data[i+1] * scale_y)) 
            for i in range(0, len(kp_data), 3)]
    box = [int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)]
    return {'classe': classe, 'box': box, 'kp': kp}

# def detect_placa(model, img, minconf=0):
#     outputs, (orig_h, orig_w) = model(img)
#     if not outputs or len(outputs) == 0:
#         return None
#     input_h, input_w = model.input_shape[2:]
#     scale_x = orig_w / input_w
#     scale_y = orig_h / input_h
#     detections = outputs[0].squeeze(0).T
#     if len(detections) == 0:
#         return None
#     best_det = None
#     best_conf = -1.0
#     for det in detections:
#         class_scores = det[4:6]
#         classe = int(np.argmax(class_scores))
#         conf = class_scores[classe]
#         if conf >= minconf and conf > best_conf:
#             best_conf = conf
#             best_det = (classe, det[6:])
#     if best_det is None:
#         return None
#     classe, kp_data = best_det
#     kp = [(int(kp_data[i] * scale_x), int(kp_data[i+1] * scale_y)) for i in range(0, len(kp_data), 3)]
#     box = [min(kp[0][0],kp[3][0]), min(kp[0][1],kp[1][1]), max(kp[1][0],kp[2][0]), max(kp[2][1],kp[3][1])]
#     return {'classe': classe, 'box': box, 'kp': kp}

def get_croped_image(resposta,img):
    x1, y1, x2, y2 = resposta['box']
    croped_img = img[y1:y2, x1:x2]
    return croped_img

def transform_keypoints(resposta):
    x1, y1, x2, y2 = resposta['box']
    return [(min(nk[0]-x1,x2-x1-1), min(nk[1]-y1,y2-y1-1)) for nk in resposta['kp']]

def warp_image(resposta, img):
    # try:
        src_points = transform_keypoints(resposta)
        width, height = 500, 200
        dst_points = [(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)]
        src = np.float32(src_points)
        dst = np.float32(dst_points)
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped_img = cv2.warpPerspective(img, matrix, (width, height))
        return warped_img
    # except:
    #     return None

def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = (gray / 255)
    return gray