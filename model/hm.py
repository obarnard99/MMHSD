import collections
import os
import time

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

# from .hm_data import HMTorchDataset, HMEvaluator, HMDataset

from .models.src.transformers.optimization import AdamW, get_linear_schedule_with_warmup

from .models.U import ModelU


# from .models.X import ModelX
# from .models.V import ModelV
# from .models.D import ModelD
# from .models.O import ModelO

# from torch.optim.swa_utils import AveragedModel, SWALR
# from torchcontrib.optim import SWA

class HM:
    def __init__(self, args):
        self.swa = args.swa

        # Select model
        if args.model == "X":
            self.model = ModelX(args)
        elif args.model == "V":
            self.model = ModelV(args)
        elif args.model == "U":
            self.model = ModelU(args)
        elif args.model == "D":
            self.model = ModelD(args)
        elif args.model == 'O':
            self.model = ModelO(args)
        else:
            raise Exception(args.model, " is not implemented.")

        # Load pre-trained weights
        self.load(args.weights)

        # Softmax
        self.logsoftmax = nn.LogSoftmax(dim=1)

        # Eval mode
        self.model.eval()

        # SWA
        if self.swa:
            self.swa_model = AveragedModel(self.model)
            self.swa_start = self.t_total * 0.75
            self.swa_scheduler = SWALR(self.optim, swa_lr=args.lr)
            self.swa_model.eval()

    def predict(self, feats, boxes, text):
        if self.swa:
            logit = self.swa_model(text, (feats, boxes))
            logit = self.logsoftmax(logit)
        else:
            logit = self.model(text, (feats, boxes))
            logit = self.logsoftmax(logit)

        score = logit[:, 1]
        _, predict = logit.max(1)

        return score, predict

    def load(self, path):
        print("Load model from %s" % path)

        state_dict = torch.load("%s" % path, map_location=torch.device('cpu'))
        new_state_dict = {}
        for key, value in state_dict.items():
            # N_averaged is a key in SWA models we cannot load, so we skip it
            if key.startswith("n_averaged"):
                print("n_averaged:", value)
                continue
            # SWA Models will start with module
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
        self.model.load_state_dict(state_dict)
