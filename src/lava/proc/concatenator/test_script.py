if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng()

    from lava.proc.io.source import RingBuffer as Source
    from lava.proc.io.sink import RingBuffer as Sink
    from lava.proc.concatenator.process import Concatenator
    from lava.proc.concatenator.models import PyConcatenatorModel
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.run_configs import Loihi2SimCfg


    num_steps = 10
    n_inputs = 3

    inp_data = [rng.integers(0, 2, size=(1, num_steps)) for _ in range(n_inputs)]

    sources = [Source(data=inp_data[i]) for i in range(n_inputs)]

    # dynamically creates a Concatenator process
    concat = Concatenator(n_inputs=n_inputs, shape=(1,))
    # creates the accompanying ProcessModel
    concat_model = PyConcatenatorModel(concat)

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

    print("Input Spikes:")
    print(np.concatenate(inp_data, axis=0))
    print("Output Spikes:")
    print(out_data)
