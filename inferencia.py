import cv2
import numpy as np
import onnxruntime as ort

class ONNXModel:
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

