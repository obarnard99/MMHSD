import requests
import json
import cv2
import numpy as np
from configs.wrapper import Config


cfg = Config('/mnt/c/Users/obarn/Projects/F-MT126-1/model/configs/main.yaml')

# load meme
text = "what separates humans from animals?"
img = cv2.imread('/mnt/c/Users/obarn/Projects/F-MT126-1/data/hmc_unseen/img/01243.png')
#img = cv2.imread('/mnt/c/Users/obarn/Downloads/test.png')
_, img_encoded = cv2.imencode('.jpg', img)

# send image for feature extraction
print('Extracting feats...')
headers = {'content-type': 'image/jpeg'}
response = requests.post('http://localhost:5000/extract', data=img_encoded.tobytes(), headers=headers)

# decode response
features = json.loads(response.text)
features['text'] = text

# send features to model
print('Processing...')
headers = {'content-type': 'application/json'}
response = requests.post('http://localhost:5001/process', json=features, headers=headers)

# decode
output = json.loads(response.text)
print(float(output['prob']))
print(float(output['label']))
