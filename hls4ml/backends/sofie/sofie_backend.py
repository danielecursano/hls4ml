import ROOT
import numpy as np

from hls4ml.backends import FPGABackend
from hls4ml.backends.sofie.sofie_utils import get_model_config, generate_sofie_model
from hls4ml.model.flow import register_flow

class SofieBackend(FPGABackend):
    def __init__(self):
        super().__init__('Sofie')
        self._default_flow = register_flow('sofie_flow', None, requires=['optimize'], backend=self.name)
        
    def create_initial_config(self, **kwargs):
        return dict()
        
    def get_default_flow(self):
        return self._default_flow
    
    def compile(self, model):
        raise NotImplementedError(f"{self.name} backend does not support compile(). To run predictions use model.predict()")
        
    def write(self, model):
        model_config = get_model_config(model)
        rmodel = generate_sofie_model(model_config)
        rmodel.Generate()
        rmodel.OutputGenerated()
        
    def predict(self, model, x):
        project_name = model.config.get_project_name()
        ROOT.gInterpreter.Declare(f'#include "{project_name}.hxx"')
        sofie_project = getattr(ROOT, f"TMVA_SOFIE_{project_name}", None)
        if not sofie_project:
            raise Exception("Sofie project not found")
        session = sofie_project.Session()
        try:
            x = np.asarray(x, dtype=np.float32)
        except Exception as e:
            raise e
        return session.infer(x)
   
