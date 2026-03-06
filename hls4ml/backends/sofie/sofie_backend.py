import ROOT
import numpy as np

from hls4ml.backends import FPGABackend
from hls4ml.model.flow import register_flow

class SofieBackend(FPGABackend):
    def __init__(self):
        super().__init__('Sofie')
        initializers = self._get_layer_initializers()
        self._default_flow = register_flow('init_layers', initializers, requires=[], backend=self.name)
        self._writer_flow = register_flow('write', ['make_stamp', 'sofie:write_hls'], requires=[self._default_flow], backend=self.name)
        # TODO add flows for optimization?
        
    def create_initial_config(self, **kwargs):
        return dict()
        
    def get_default_flow(self):
        return self._default_flow
            
    def get_writer_flow(self):
        return self._writer_flow
        
    def compile(self, model):
        raise NotImplementedError(f"{self.name} backend does not support compile(). To run predictions use model.predict()")
    
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
        
