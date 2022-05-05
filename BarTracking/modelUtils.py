import os
import json
import onnxruntime as ort

with open(os.path.join(os.path.dirname(__file__), 'model_info.json'), 'r') as f:
    model_info = json.load(f)

model_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in os.listdir(os.path.dirname(__file__)) if fn.endswith('.onnx')]

get_session = lambda: ort.InferenceSession(model_paths[0], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# test
if __name__ == '__main__':
    print(model_paths)
    print(model_info)