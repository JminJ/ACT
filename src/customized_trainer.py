from argparse import ArgumentError, Namespace
from transformers import Trainer, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from overrides import overrides

class ActTrainer(Trainer):
    def __init__(self, args:Namespace, extracted_data:dict):
        super(ActTrainer, self).__init__()
        self.args = args
        self.extracted_data = extracted_data
        ## weight applied loss define
        if self.args.use_label_weight:
            self.weighted_loss = self.applied_weight_loss_define()
        else:
            self.weighted_loss = None

    @overrides
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if self.args.use_label_weight:
                loss = self.calc_weight_loss(outputs, labels=labels)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def applied_weight_loss_define(self):
        loss_weight = self.calc_label_weight(self.extracted_data['label_cnts'])
        weighted_loss = nn.CrossEntropyLoss(weight = loss_weight)

        return weighted_loss

    def calc_weight_loss(self, outputs, labels):
        logits = list(outputs['logits'])

        loss_val = self.weighted_loss(logits.detach(), labels.detach())

        return loss_val
    
    def calc_label_weight(self, label_cnts:list):
        loss_weight = [max(label_cnts) / w for w in label_cnts]

        tensor_loss_weight = torch.tensor(loss_weight, dtype=torch.float)
        print(f'tenosr_loss_weight : {tensor_loss_weight}')

        return tensor_loss_weight
