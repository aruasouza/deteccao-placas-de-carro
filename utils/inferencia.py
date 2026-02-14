import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

class ONNXYOLO:
    def __init__(self, path):
        self.session = ort.InferenceSession(path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
    def predict(self, img):
        h, w = self.input_shape[2:]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_prep = cv2.resize(img_rgb, (w, h)).astype(np.float32) / 255.0
        img_prep = np.transpose(img_prep, (2, 0, 1))[np.newaxis, ...]
        outputs = self.session.run(None, {self.input_name: img_prep})
        return outputs, img.shape[:2]
    
    def __call__(self, *args, **kwds):
        return self.predict(*args, **kwds)

class ONNXCharModel:
    def __init__(self, path):
        self.session = ort.InferenceSession(path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
    
    # def predict(self, imgs):
    #     h, w = self.input_shape[2:]
    #     batch = []
    #     for img in imgs:
    #         img_prep = cv2.resize(img, (w, h)).astype(np.float32)
    #         img_norm = (img_prep - 0.5) / 0.5
    #         img_norm = img_norm[np.newaxis, ...]
    #         batch.append(img_norm)
    #     batch = np.array(batch)
    #     outputs = self.session.run(None, {self.input_name: batch})
    #     return outputs
    
    def predict(self, imgs):
        h, w = self.input_shape[2:]
        batch = []
        for img in imgs:
            img_uint8 = (img * 255).astype(np.uint8)
            
            pil_img = Image.fromarray(img_uint8, mode='L')
            pil_img = pil_img.resize((w, h), Image.BILINEAR)
            
            img_array = np.array(pil_img).astype(np.float32) / 255.0
            img_norm = (img_array - 0.5) / 0.5
            img_norm = img_norm[np.newaxis, ...]
            batch.append(img_norm)
        
        batch = np.array(batch)
        outputs = self.session.run(None, {self.input_name: batch})
        return outputs

    
    def __call__(self, *args, **kwds):
        return self.predict(*args, **kwds)