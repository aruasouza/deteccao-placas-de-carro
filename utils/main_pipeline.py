import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils.me_pipeline as me_pipeline
import utils.br_pipeline as br_pipeline
from utils.inferencia import ONNXYOLO

model = ONNXYOLO('onnx_models/best.onnx')

def extraction_pipeline(img, model=model, onnx = True):
    if onnx:
        deteccao = detect_placa_onnx(model, img)
    else:
        deteccao = detect_placa(model, img)
    if deteccao is None:
        return None
    gray = grayscale(img)
    croped_img = get_croped_image(deteccao, gray)
    warped_img = warp_image(deteccao, croped_img)
    if deteccao['classe'] == 1:
        final_output = br_pipeline.pipeline(warped_img, 30)
    else:
        final_output = me_pipeline.pipeline(warped_img, 35)
    return final_output

def detect_placa(model,img):
    results = model(img)
    if not results:
        return None
    res = results[0]
    try:
        classe = round(float(res.boxes.cls.numpy()[0]))
        box = [round(x) for x in res.boxes.xyxy.numpy()[0]]
        kp = [(round(x),round(y)) for x,y in res.keypoints.xy.numpy()[0]]
        return {'classe':classe, 'box': box, 'kp':kp}
    except:
        return None

def detect_placa_onnx(model, img):
    outputs, (orig_h, orig_w) = model(img)
    if not outputs or len(outputs) == 0:
        return None
    input_h, input_w = model.input_shape[2:]
    scale_x = orig_w / input_w
    scale_y = orig_h / input_h
    detections = outputs[0][0]
    if len(detections) == 0:
        return None
    best_det = detections[0]
    x1, y1, x2, y2 = best_det[:4]
    conf = best_det[4]
    classe = int(best_det[5])
    kp_data = best_det[6:]
    kp = [(int(kp_data[i] * scale_x), int(kp_data[i+1] * scale_y)) 
            for i in range(0, len(kp_data), 3)]
    box = [int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)]
    return {'classe': classe, 'box': box, 'kp': kp}

def get_croped_image(resposta,img):
    x1, y1, x2, y2 = resposta['box']
    croped_img = img[y1:y2, x1:x2]
    return croped_img

def transform_keypoints(resposta):
    x1, y1, x2, y2 = resposta['box']
    return [(min(nk[0]-x1,x2-x1-1), min(nk[1]-y1,y2-y1-1)) for nk in resposta['kp']]

def warp_image(resposta, img):
    src_points = transform_keypoints(resposta)
    width, height = 500, 200
    dst_points = [(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)]
    src = np.float32(src_points)
    dst = np.float32(dst_points)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(img, matrix, (width, height))
    return warped_img

def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = (gray / 255)
    return gray

def draw_results(img, resposta):
    img_result = draw_bounding_box(img, resposta)
    img_result = draw_keypoints(img_result, resposta)
    return img_result

def draw_keypoints(img, resposta, color=(0, 0, 255), radius=3):
    img_with_keypoints = img.copy()
    keypoints = resposta['kp']
    for kp in keypoints:
        x, y = kp
        cv2.circle(img_with_keypoints, (int(x), int(y)), radius, color, -1)
    return img_with_keypoints

def draw_bounding_box(img,resposta):
    result_image = img.copy()
    x1, y1, x2, y2 = resposta['box']
    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return result_image

def plot_extracted_chars(extracted):
    letras = extracted['letras']
    numeros = extracted['numeros']
    total = len(letras) + len(numeros)
    fig, axes = plt.subplots(1, total, figsize=(total * 2, 3))
    if total == 1:
        axes = [axes]
    for i, letra in enumerate(letras):
        axes[i].imshow(letra, cmap='gray')
        axes[i].set_title(f'Letra {i+1}')
        axes[i].axis('off')
    for i, numero in enumerate(numeros):
        axes[len(letras) + i].imshow(numero, cmap='gray')
        axes[len(letras) + i].set_title(f'Número {i+1}')
        axes[len(letras) + i].axis('off')
    plt.tight_layout()
    plt.show()

def show_image(img,color = False):
    if color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.show()