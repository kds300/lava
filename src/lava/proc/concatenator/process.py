"""Dynamically Created Concatenator Process

`Concatenator` class dynamically creates a new class with the requested
number of input ports. The concatenator takes the inputs and
concatenates them into a single output.

"""

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort


class AbstractConcatenator(AbstractProcess):
    """Abstract class for a concatenator"""
    x_in: dict[int, InPort]
    x_out: OutPort
    n_inputs: int
    shape: tuple[int]

    def __init__(self, shape):
        super().__init__(shape=shape)

        for i in range(self.n_inputs):
            setattr(self, f"_x_in_{i}", InPort(shape=shape))

        self.x_in = {
            i: getattr(self, f"_x_in_{i}") for i in range(self.n_inputs)
        }
        self.x_out = OutPort(shape=(self.n_inputs,))


class Concatenator:
    """Create a Concatenator process with requested number of inputs."""
    def __new__(cls, n_inputs, shape):
        """
        Parameters
        ----------
        n_inputs: int
            Number of InPorts to add to the concatenator.
        shape: tuple
            Shape of the InPorts. OutPort will have shape (n_inputs, *shape)
        """
        attrs = {'n_inputs': n_inputs}
        new_proc = type(
            f'Concatenator_{n_inputs}',
            (AbstractConcatenator,),
            attrs
        )

        return new_proc(shape=shape)
