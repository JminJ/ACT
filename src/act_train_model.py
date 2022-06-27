import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from act_collate_fn import ActCollateFN
from act_dataset import ActDataset
from customized_trainer import ActTrainer
from setting_for_target_dataset import ExtractArgsFromData







def __main__():
    parser = argparse.ArgumentParser(description="ACT(Automatic Classifier Trainer)")
    parser.add_argument('')