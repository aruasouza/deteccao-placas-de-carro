import cv2
import matplotlib.pyplot as plt

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