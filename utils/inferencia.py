import cv2
import numpy as np
from PIL import Image
import os

env = os.getenv('ENV', 'DEV')

if env == 'PROD':
    from rknnlite.api import RKNNLite as RKNN
else:
    import onnxruntime as ort

class ONNXYOLO:
    def __init__(self, path):
        if env == 'CUDA':
            import torch
            self.session = ort.InferenceSession(path,providers=["CUDAExecutionProvider"])
        else:
            self.session = ort.InferenceSession(path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
    def predict(self, img):
        h, w = self.input_shape[2:]
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_image, mode='RGB')
        pil_img = pil_img.resize((w, h), Image.BILINEAR)
        img_array = np.array(pil_img).astype(np.float32) / 255.0
        img_prep = np.transpose(img_array, (2, 0, 1))[np.newaxis, ...]
        outputs = self.session.run(None, {self.input_name: img_prep})
        return outputs, img.shape[:2]
    
    def __call__(self, *args, **kwds):
        return self.predict(*args, **kwds)
    
class ONNXCharModel:
    def __init__(self, path):
        if env == 'CUDA':
            import torch
            self.session = ort.InferenceSession(path,providers=["CUDAExecutionProvider"])
        else:
            self.session = ort.InferenceSession(path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
    
    def predict(self, imgs):
        h, w = self.input_shape[2:]
        results = []
        for img in imgs:
            img_uint8 = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8, mode='L')
            pil_img = pil_img.resize((w, h), Image.BILINEAR)
            img_array = np.array(pil_img).astype(np.float32) / 255.0
            img_norm = (img_array - 0.5) / 0.5
            img_norm = img_norm[np.newaxis, np.newaxis, ...]
            output = self.session.run(None, {self.input_name: img_norm})
            results.append(output[0])
        
        return [np.concatenate(results, axis=0)]
    
    def __call__(self, *args, **kwds):
        return self.predict(*args, **kwds)

class RKNNYOLO:
    def __init__(self, path):
        self.rknn = RKNN()
        self.rknn.load_rknn(path)
        self.rknn.init_runtime()
        self.input_shape = (1, 3, 640, 640)
        
    def predict(self, img):
        h, w = self.input_shape[2:]
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_image, mode='RGB')
        pil_img = pil_img.resize((w, h), Image.BILINEAR)
        img_array = np.array(pil_img,'uint8')
        img_prep = img_array[np.newaxis, ...]
        outputs = self.rknn.inference(inputs=[img_prep])
        return outputs, img.shape[:2]
    
    def __call__(self, *args, **kwds):
        return self.predict(*args, **kwds)

class RKNNCharModel:
    def __init__(self, path):
        self.rknn = RKNN()
        self.rknn.load_rknn(path)
        self.rknn.init_runtime()
        self.input_shape = (1, 1, 28, 28)
    
    def predict(self, imgs):
        h, w = self.input_shape[2:]
        results = []
        for img in imgs:
            img_uint8 = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8, mode='L')
            pil_img = pil_img.resize((h, w), Image.BILINEAR)
            img_array = np.array(pil_img,'uint8')
            img_norm = img_array[np.newaxis, np.newaxis, ...]
            output = self.rknn.inference(inputs=[img_norm])
            results.append(output[0])
        return [np.concatenate(results, axis=0)]
    
    def __call__(self, *args, **kwds):
        return self.predict(*args, **kwds)
