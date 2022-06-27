from argparse import ArgumentError, Namespace
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
    * focal_alpha   x
    * focal_gamma   x

    ---
    * board type -> wandb or tensorboard
    * using_model -> default : bert.
    * device -> default : cpu.
    * text_column 
    * model_save_path -> default : "./../mdoel_save"

'''

class ExtractArgsFromData:
    def __init__(self, args:Namespace):
        self.args = args
        self.dataset_path = self.args.dataset_path
        self.target_dataset = self.read_data()

    ## 메인 함수
    def extract(self):
        target_unique, label_cnts = self.get_unique(self.target_dataset)

        return {
            'target_unique' : list(target_unique),
            'label_cnts' : list(label_cnts)
        }

    ## 현재 dataset을 반환.
    def return_temp_dataset(self):
        return self.target_dataset

    # 파일 포맷에 맞게 데이터 로드
    def read_data(self):
        file_extension = self.dataset_path.split('.')[-1]
        if file_extension == 'xlsx':
            if self.args.sheet_name == None:
                raise ArgumentError('sheet_name argument is None!') 
            target_dataset = pd.read_excel(self.dataset_path, sheet_name=self.args.sheet_name)

        elif file_extension == 'csv':
            target_dataset = pd.read_csv(self.dataset_path)

        elif file_extension == 'tsv':
            target_dataset = pd.read_csv(self.dataset_path, sep = '\t')

        else:
            raise ArgumentError('...')

        return target_dataset

    def get_unique(self, target_dataset):
        try:
            dataset_unique, label_cnts = np.unique(target_dataset.loc[:, self.args.target_column], return_counts = True)
        except:
            raise TypeError(f'datas in {self.args.target_column} column, need to convert Int.')
        print(f'target_column : {self.args.target_column}')
        print(f'dataset_unique : {dataset_unique}')
        print(f'label_cnts : {label_cnts}')

        return dataset_unique, label_cnts
