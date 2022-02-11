import transformers
import timm

class GetSelectedBaseModel:
    r'''
        Get huggingface Pretrained Model user selected.
        Can use all huggingface's ...Model parameters by set args parameter.


        Arg:
            select_model_name (`str`, *default*):
                Use to Classifier's base model. Users can select Model name both ['electra', 'bart'].

            user_model (`str`, *optional*, defaults to None):
                If users have pretrained language model, can use it by define this parameter.
                ex) user_model = '.../.../electraPretrain' 
        '''
    def __init__(self, select_model_name, user_model=None):
        self.select_model_name = select_model_name
        self.user_model = user_model

    def init_selected_model(self):
        # initialize user selected model
        if self.select_model_name == 'electra':
            if self.user_model is not None:
                model = transformers.ElectraModel.from_pretrained(self.user_model)
            else:
                model = transformers.ElectraModel.from_pretrained()
        elif self.select_model_name == 'bart':
            pass

    def get_selected_model(self):
        selected_model = self.init_selected_model(self)
        
        return selected_model