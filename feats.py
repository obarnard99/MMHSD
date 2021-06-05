from flask import Flask, request
import numpy as np
import cv2
from configs.wrapper import Config
from feats.extract import Detector

app = Flask(__name__)
cfg = Config('/mnt/c/Users/obarn/Projects/F-MT126-1/model/configs/main.yaml')
detector = Detector(cfg)


@app.route('/extract', methods=['POST'])
def handler():
    # decode image
    nparr = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    features = detector.doit(img)
    obj_num = features['num_boxes']
    feats = features['features'].copy()
    boxes = features['boxes'].copy()
    assert obj_num == len(boxes) == len(feats)

    # Normalize the boxes (to 0 ~ 1)
    img_h, img_w = features['img_h'], features['img_w']

    if cfg.num_pos == 5:
        # For DeVLBert taken from VilBERT
        image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
        image_location[:, :4] = boxes
        image_location[:, 4] = (image_location[:, 3] - image_location[:, 1]) * (
                image_location[:, 2] - image_location[:, 0]) / (float(img_w) * float(img_h))
        boxes = image_location

    boxes = boxes.copy()
    boxes[:, (0, 2)] /= img_w
    boxes[:, (1, 3)] /= img_h
    np.testing.assert_array_less(boxes, 1 + 1e-5)
    np.testing.assert_array_less(-boxes, 0 + 1e-5)

    if cfg.num_pos == 6:
        # Add width & height
        width = (boxes[:, 2] - boxes[:, 0]).reshape(-1, 1)
        height = (boxes[:, 3] - boxes[:, 1]).reshape(-1, 1)

        boxes = np.concatenate((boxes, width, height), axis=-1)

        # In UNITER they use 7 Pos Feats (See _get_img_feat function in their repo)
        if cfg.model == "U":
            boxes = np.concatenate([boxes, boxes[:, 4:5] * boxes[:, 5:]], axis=-1)

    new_feats = {'boxes': boxes.tolist(), 'feats': feats.tolist()}
    return new_feats


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000)
