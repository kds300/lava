"""Process models for the Concatenator.

`PyConcatenatorModel` takes a `Concatenator` process as input and returns
a process model with the required number of InPorts.

"""

import numpy as np

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import requires


@requires(CPU)
class PyAbstractScalarConcatenatorModel(PyLoihiProcessModel):
    x_in: dict[int, PyInPort]
    x_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    n_inputs: int

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.shape = proc_params.get("shape", (1,))
        self.concat_method = np.concatenate

    def run_spk(self):
        _inp_data = np.zeros(self.x_out.shape)
        _inp_data = self.concat_method(
            [getattr(self, f"_x_in_{i}").recv() for i in range(self.n_inputs)],
            axis=0
        )
        self.x_out.send(_inp_data)

@requires(CPU)
class PyAbstractVectorConcatenatorModel(PyAbstractScalarConcatenatorModel):
    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.concat_method = np.stack

class PyScalarConcatenatorModel:
    def __new__(cls, proc):

        model_attrs = {'n_inputs': proc.n_inputs}
        for i in range(proc.n_inputs):
            model_attrs[f"_x_in_{i}"] = LavaPyType(PyInPort.VEC_DENSE, float)

        new_model = type(
            f'PyConcatenatorModel_{proc.n_inputs}',
            (PyAbstractScalarConcatenatorModel,),
            model_attrs
        )

        # set up implementation details
        new_model.implements_process = type(proc)
        new_model.implements_protocol = LoihiProtocol

        return new_model

class PyVectorConcatenatorModel:
    def __new__(cls, proc):

        model_attrs = {'n_inputs': proc.n_inputs}
        for i in range(proc.n_inputs):
            model_attrs[f"_x_in_{i}"] = LavaPyType(PyInPort.VEC_DENSE, float)

        new_model = type(
            f'PyConcatenatorModel_{proc.n_inputs}',
            (PyAbstractVectorConcatenatorModel,),
            model_attrs
        )

        # set up implementation details
        new_model.implements_process = type(proc)
        new_model.implements_protocol = LoihiProtocol

        return new_model