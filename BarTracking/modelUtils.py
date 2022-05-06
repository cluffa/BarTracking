import os
import json
import onnxruntime as ort

#with open(os.path.join(os.path.dirname(__file__), 'model_info.json'), 'r') as f:
#    model_info = json.load(f)

base_path = os.path.dirname(__file__)
model_names = [fn for fn in os.listdir(os.path.dirname(__file__)) if fn.endswith('.onnx')]

# test
if __name__ == '__main__':
    print(base_path)
    print(model_names)