from argparse import ArgumentError
import pandas as pd
import numpy as np

'''
args
    * weight decay
    * learning rate decay(schedular)
    * fp16
    * loss_type -> use_focal
    * sheet_name
    * target_column

    * use_label_weight
    * focal_alpha
    * focal_gamma
'''

class ExtractArgsFromData:
    def __init__(self, args):
        self.args = args
        self.dataset_path = self.args.dataset_path
        self.target_dataset = self.read_data()

    def extract(self):
        target_unique, label_cnts = self.get_unique()


    def read_data(self):
        file_extension = self.dataset_path.split('.')[-1]
        if file_extension == 'xlsx':
            if self.args.sheet_name == None:
                raise ArgumentError('if you use excel file, ...') # 이따 기입
            target_dataset = pd.read_excel(self.dataset_path, sheet_name=self.args.sheet_name)

        elif file_extension == 'csv':
            target_dataset = pd.read_csv(self.dataset_path)

        elif file_extension == 'tsv':
            target_dataset = pd.read_csv(self.dataset_path, sep = '\t')

        else:
            raise ArgumentError('...')

    def get_unique(self):
        try:
            dataset_unique, label_cnts = np.unique(self.target_dataset.loc[:, self.args.target_column], return_counts = True)
        except:
            raise TypeError(f'datas in {self.args.target_column} column, need to convert Int.')
        print(f'target_column : {self.args.target_column}')
        print(f'dataset_unique : {dataset_unique}')
        print(f'label_cnts : {label_cnts}')


        return dataset_unique, label_cnts
