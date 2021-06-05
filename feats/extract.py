# coding=utf-8
# Copyleft 2019 Project LXRT

import csv
import os
import sys

csv.field_size_limit(sys.maxsize)

# import some common libraries
import numpy as np
import torch

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs

from torchvision.ops import nms
from detectron2.structures import Boxes, Instances

D2_ROOT = os.path.dirname(os.path.dirname(detectron2.__file__))  # Root of detectron2


def fast_rcnn_inference_single_image(
        boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Select max scores
    max_scores, max_classes = scores.max(1)  # R x C --> R
    num_objs = boxes.size(0)
    boxes = boxes.view(-1, 4)
    idxs = torch.arange(num_objs) * num_bbox_reg_classes + max_classes  # Removed .cuda() here
    max_boxes = boxes[idxs]  # Select max boxes according to the max scores.

    # Apply NMS
    keep = nms(max_boxes, max_scores, nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores = max_boxes[keep], max_scores[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = max_classes[keep]

    return result, keep


class Detector:
    def __init__(self, config):
        self.attr = config.attr
        self.batchsize = 1
        self.MIN_BOXES = config.num_feats
        self.MAX_BOXES = config.num_feats

        # Build model and load weights.
        cfg = get_cfg()  # Renew the cfg file
        if self.attr:
            cfg.merge_from_file(os.path.join(
                D2_ROOT, "configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml"))
        else:
            cfg.merge_from_file(os.path.join(
                D2_ROOT, "configs/VG-Detection/faster_rcnn_R_101_C4_caffemaxpool.yaml"))
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
        cfg.INPUT.MIN_SIZE_TEST = 600
        cfg.INPUT.MAX_SIZE_TEST = 1000
        cfg.MODEL.RPN.NMS_THRESH = 0.7
        cfg.MODEL.DEVICE = 'cpu'

        if self.attr:
            cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl"
        else:
            cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"

        self.detector = DefaultPredictor(cfg)

    def doit(self, raw_image):
        raw_images = [raw_image]  # TODO: Hacky implementation
        with torch.no_grad():
            # Preprocessing
            inputs = []
            for raw_image in raw_images:
                image = self.detector.transform_gen.get_transform(raw_image).apply_image(raw_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                inputs.append({"image": image, "height": raw_image.shape[0], "width": raw_image.shape[1]})
            images = self.detector.model.preprocess_image(inputs)

            # Run Backbone Res1-Res4
            features = self.detector.model.backbone(images.tensor)

            # Generate proposals with RPN
            proposals, _ = self.detector.model.proposal_generator(images, features, None)

            # Run RoI head for each proposal (RoI Pooling + Res5)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            features = [features[f] for f in self.detector.model.roi_heads.in_features]
            box_features = self.detector.model.roi_heads._shared_roi_transform(
                features, proposal_boxes
            )
            feature_pooled = box_features.mean(dim=[2, 3])  # (sum_proposals, 2048), pooled to 1x1

            # Predict classes and boxes for each proposal.
            if self.attr:
                pred_class_logits, pred_attr_logits, pred_proposal_deltas = self.detector.model.roi_heads.box_predictor(
                    feature_pooled)
            else:
                pred_class_logits, pred_proposal_deltas = self.detector.model.roi_heads.box_predictor(feature_pooled)

            rcnn_outputs = FastRCNNOutputs(
                self.detector.model.roi_heads.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                self.detector.model.roi_heads.smooth_l1_beta,
            )

            ####
            if self.attr:
                attr_prob = pred_attr_logits[..., :-1].softmax(-1)
                max_attr_prob, max_attr_label = attr_prob.max(-1)

            # Fixed-number NMS
            instances_list, ids_list = [], []
            probs_list = rcnn_outputs.predict_probs()
            boxes_list = rcnn_outputs.predict_boxes()
            for probs, boxes, image_size in zip(probs_list, boxes_list, images.image_sizes):
                for nms_thresh in np.arange(0.3, 1.0, 0.1):
                    instances, ids = fast_rcnn_inference_single_image(
                        boxes, probs, image_size,
                        score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=self.MAX_BOXES
                    )
                    if len(ids) >= self.MIN_BOXES:
                        break
                #####
                if self.attr:
                    max_attr_prob = max_attr_prob[ids].detach()
                    max_attr_label = max_attr_label[ids].detach()

                    instances.attr_scores = max_attr_prob
                    instances.attr_classes = max_attr_label

                instances_list.append(instances)
                ids_list.append(ids)

            # Post processing for features
            features_list = feature_pooled.split(
                rcnn_outputs.num_preds_per_image)  # (sum_proposals, 2048) --> [(p1, 2048), (p2, 2048), ..., (pn, 2048)]
            roi_features_list = []

            for ids, features in zip(ids_list, features_list):
                roi_features_list.append(features[ids].detach())

            # Post processing for bounding boxes (rescale to raw_image)
            raw_instances_list = []
            for instances, input_per_image, image_size in zip(
                    instances_list, inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                raw_instances = detector_postprocess(instances, height, width)
                raw_instances_list.append(raw_instances)

            for instances, features in zip(instances_list, roi_features_list):
                num_objects = len(instances)
                item = {
                    "img_h": raw_image.shape[0],
                    "img_w": raw_image.shape[1],
                    "objects_id": instances.pred_classes.numpy(),  # int64
                    "objects_conf": instances.scores.numpy(),  # float32
                    "num_boxes": num_objects,
                    "boxes": instances.pred_boxes.tensor.numpy(),  # float32
                    "features": features.numpy()  # float32
                }

                if self.attr:
                    item["attrs_id"] = instances.attr_classes.numpy()  # int64
                    item["attrs_conf"] = instances.attr_scores.numpy()  # float32
                else:
                    item["attrs_id"] = np.zeros(num_objects, np.int64)  # int64
                    item["attrs_conf"] = np.zeros(num_objects, np.float32)  # float32

            return item
