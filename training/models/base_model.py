import os
import json
import numpy as np


class BaseModel():
    
    def __init__(self):
        self.model = None
        self.num_classes = None
        self.model_params = {}
        self.dimensions = None
        self.priors = None


    def create_model(self):
        pass
    

    def predict(self, X, argmax=True):

        y_pred = self.model.predict(X) # (num_examples, num_outputs)
        if argmax:
            y_pred = np.argmax(y_pred, axis=-1) # (num_examples,)
        return y_pred
    
    
    def set_skopt_dimensions(self):
        pass
    
    
    def load_model(self, dirpath):

        weightspath = os.path.join(dirpath,'model.h5')
        self.model.load_weights(weightspath)


    def save_model(self, dirpath):

        jsonpath = os.path.join(dirpath,'model.json')
        weightspath = os.path.join(dirpath,'model.h5')
        model_dict = json.loads(self.model.to_json())
        with open(jsonpath, "w") as json_file:
            json_file.write( json.dumps(model_dict,indent=4) )
        self.model.save_weights(weightspath)
    
    
    
    
    
    
    
    
    
    