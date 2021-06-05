from flask import Flask, request
import torch
from configs.wrapper import Config
from model import HM

app = Flask(__name__)
cfg = Config('/mnt/c/Users/obarn/Projects/F-MT126-1/model/configs/main.yaml')
hm = HM(cfg)


@app.route('/process', methods=['POST'])
def handler():
    data = request.json
    boxes = torch.tensor(data['boxes'])
    feats = torch.tensor(data['feats'])
    text = data['text']
    prob, label = hm.predict([feats], [boxes], [text])  # Lists needed as predict can take multiple memes
    response = {'prob': str(prob.detach().numpy()[0]), 'label': str(label.detach().numpy()[0])}
    return response


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5001)


