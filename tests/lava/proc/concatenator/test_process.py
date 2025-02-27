"""Unittests for Concatenator Process

"""

import unittest

import numpy as np

from lava.proc.io.source import RingBuffer as Source
from lava.proc.io.sink import RingBuffer as Sink
from lava.proc.concatenator.process import Concatenator
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg


# setup RNG
rng = np.random.default_rng()


class TestScalarConcatenator(unittest.TestCase):
    """Unit tests for the Concatenator process when inputs are scalars."""
    def test_concatenate_cpu(self):
        """Test concatenation of inputs on CPU."""
        num_steps = 100
        n_inputs = 3

        inp_data = [rng.integers(0, 2, size=(1, num_steps)) for _ in range(n_inputs)]

        sources = [Source(data=inp_data[i]) for i in range(n_inputs)]

        # dynamically creates a Concatenator process and model
        concat, concat_model = Concatenator(
            n_inputs=n_inputs, shape=(1,))

        sink = Sink(shape=concat.x_out.shape, buffer=num_steps)

        for i in range(n_inputs):
            sources[i].s_out.connect(concat.x_in[i])
        concat.x_out.connect(sink.a_in)

        concat.run(
            condition=RunSteps(num_steps=num_steps),
            run_cfg=Loihi2SimCfg(
                select_tag='fixed_pt',
                # required for compiler to know which model to use
                exception_proc_model_map={concat: concat_model}
            )
        )
        out_data = sink.data.get().astype(int)
        concat.stop()

        for step in range(num_steps):
            # print(f"Step {step}:")
            # print("\tInput Spikes:")
            # for i in range(n_inputs):
                # print(f"\t\tFrom Source {i}: {inp_data[i][:, step]}")
            # print("\tExpected Output:")

            expected_output = np.concatenate([[inp_data[i][:, step]] for i in range(n_inputs)], axis=0)

            # print(expected_output)

            # print("\tOutput Spikes:")
            # print(out_data[:, :, step])
            # print("")

            self.assertTrue(
                np.array_equal(expected_output, out_data[:, :, step]),
                msg=f"Input:\n{inp_data}\nOutput:\n{out_data}"
            )

class TestVectorConcatenator(unittest.TestCase):
    """Unit tests for the Concatenator process when inputs are arrays."""
    def test_concatenate_cpu(self):
        """Test concatenation of inputs on CPU."""
        num_steps = 100
        num_values = 2
        n_inputs = 3

        inp_data = [rng.uniform(-1, 1, size=(num_values, num_steps)) for _ in range(n_inputs)]

        sources = [Source(data=inp_data[i]) for i in range(n_inputs)]

        # dynamically creates a Concatenator process and model
        concat, concat_model = Concatenator(
            n_inputs=n_inputs, shape=(num_values,), mode='stack')

        sink = Sink(shape=concat.x_out.shape, buffer=num_steps)

        for i in range(n_inputs):
            sources[i].s_out.connect(concat.x_in[i])
        concat.x_out.connect(sink.a_in)

        concat.run(
            condition=RunSteps(num_steps=num_steps),
            run_cfg=Loihi2SimCfg(
                select_tag='floating_pt',
                # required for compiler to know which model to use
                exception_proc_model_map={concat: concat_model}
            )
        )

        out_data = sink.data.get().astype(float)
        concat.stop()

        # inp_data = np.concatenate(inp_data, axis=0)

        for step in range(num_steps):
            # print(f"Step {step}:")
            # print("\tInput Spikes:")
            # for i in range(n_inputs):
            #     print(f"\t\tFrom Source {i}: {inp_data[i][:, step]}")
            # print("\tExpected Output:")

            expected_output = np.stack([inp_data[i][:, step] for i in range(n_inputs)], axis=0)

            # print(expected_output)
            # print("\tOutput Spikes:")
            # print(out_data[:, :, step])
            # print("")

            self.assertTrue(
                np.array_equal(expected_output, out_data[:, :, step]),
                msg=f"Input:\n{inp_data}\nOutput:\n{out_data}"
            )

if __name__ == "__main__":
    unittest.main()
