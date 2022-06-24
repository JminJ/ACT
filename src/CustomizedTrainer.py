from argparse import ArgumentError, Namespace
from transformers import Trainer, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class ActTrainer(Trainer):
    def __init__(self, args:Namespace, extracted_data:dict):
        super(ActTrainer, self).__init__()
        self.args = args
        self.extracted_data = extracted_data

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
                loss = self.calc_loss_applied_weight(outputs, labels)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def calc_loss_applied_weight(self, outputs, labels):
        logits = list(outputs['logits'])
    
        loss_weight = self.calc_label_weight(self.extracted_data['label_cnts'])
        loss = nn.CrossEntropyLoss(weight = loss_weight)

        ## calc loss
        loss_val = loss(logits.detach(), labels.detach())

        return loss_val
    
    def calc_label_weight(self, label_cnts:list):
        loss_weight = [max(label_cnts) / w for w in label_cnts]

        tensor_loss_weight = torch.tensor(loss_weight, dtype=torch.float)
        print(f'tenosr_loss_weight : {tensor_loss_weight}')

        return tensor_loss_weight
