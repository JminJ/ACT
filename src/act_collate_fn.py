from argparse import ArgumentError
from transformers import AutoTokenizer

class ActCollateFN:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.using_model)
        self.check_text_column()

    def __call__(self, batch):
        toked_output = self.tokenizer.encode_plus(
            batch[self.args.text_column],
            padding = 'longest',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        batch['toked_output'] = toked_output

        return batch
        
    def check_text_column(self):
        if self.args.text_column is None:
            raise ArgumentError('Please, define "--text_column" argument.')