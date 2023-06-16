import json,cv2, requests
import numpy as np

im = cv2.imread('test.png')
blob = cv2.dnn.blobFromImage(im, 1/ 255.0, (640, 640), (0,0,0), swapRB=True, crop=False)
im = cv2.resize(im, (640, 640)).reshape(1,640,640,3)
data = json.dumps({"instances": blob.tolist()})
data1 = json.dumps({"instances": im.tolist()})
headers = {"content-type": "application/json"}
response = requests.post("http://localhost:8501/v1/models/carafe/versions/1:predict", data=data, headers=headers)
content = response.content.decode('utf-8')
results = np.array(json.loads(content)['predictions'][0]["output0"])
print(f"results.shape: {results.shape}")
response = requests.post("http://localhost:8501/v1/models/carafe/labels/canary:predict", data=data, headers=headers)
content = response.content.decode('utf-8')
results = np.array(json.loads(content)['predictions'][0]["output0"])
print(f"results.shape: {results.shape}")
response = requests.post("http://localhost:8501/v1/models/bw/versions/22:predict", data=data1, headers=headers)
content = response.content.decode('utf-8')
results = np.array(json.loads(content)['predictions'][0])
print(f"results.shape: {results.shape}")