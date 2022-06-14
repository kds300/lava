# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from abc import abstractmethod
import logging
import typing as ty

import numpy as np

from lava.magma.compiler.channels.pypychannel import (
    CspSelector,
    CspRecvPort,
    CspSendPort
)
from lava.magma.core.sync.protocol import AbstractSyncProtocol
from lava.magma.runtime.mgmt_token_enums import (
    enum_to_np,
    enum_equal,
    MGMT_RESPONSE,
    MGMT_COMMAND,
)
from lava.magma.runtime.runtime_services.enums import (
    LoihiPhase,
    LoihiVersion
)
from lava.magma.runtime.runtime_services.interfaces import \
    AbstractRuntimeService

try:
    from nxsdk.arch.base.nxboard import NxBoard
except(ImportError):
    class NxBoard():
        pass

"""The RuntimeService interface is responsible for
coordinating the execution of a group of process models belonging to a common
synchronization domain. The domain will follow a SyncProtocol or will be
asynchronous. The processes and their corresponding process models are
selected by the Runtime dependent on the RunConfiguration assigned at the
start of execution. For each group of processes which follow the same
protocol and execute on the same node, the Runtime creates a RuntimeService.
Each RuntimeService coordinates all actions and commands from the Runtime,
 transmitting them to the the processes under it's managment and
returning action and command responses back to Runtime.

RuntimeService Types:

1. PyRuntimeService: (Abstract Class) Coordinates process models executing on
   the CPU and written in Python.
   Concrete Implementations:
    a. LoihiPyRuntimeService: Coordinates process models executing on
       the CPU and written in Python and following the LoihiProtocol.
    b. AsyncPyRuntimeService: Coordinates process models executing on
       the CPU and written in Python and following the AsyncProtocol.

2. CRuntimeService: (Abstract Class) Coordinates/Manages process models
   executing on the CPU/Embedded CPU and written in C
   Concrete Implementations:
    a. LoihiCRuntimeService: Coordinates process models executing on
       the CPU/Embedded CPU and written in C and following the LoihiProtocol.
3. NcRuntimeService: (Abstract Class) Coordinates/Manages process models
   executing on a Loihi NeuroCore.
   Concrete Implementations:
    a. NxSdkRuntimeService: Coordinates process models executing on a Loihi
       NeuroCore and written in Python following the LoihiProtocol.
"""


class PyRuntimeService(AbstractRuntimeService):
    """Abstract RuntimeService for Python, it provides base methods
    for start and run. It is not meant to instantiated directly
    but used by inheritance
    """

    def __init__(self,
                 protocol: ty.Type[AbstractSyncProtocol],
                 loglevel: int = logging.WARNING,):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(loglevel)
        super(PyRuntimeService, self).__init__(
            protocol=protocol
        )
        self.service_to_process: ty.Iterable[CspSendPort] = []
        self.process_to_service: ty.Iterable[CspRecvPort] = []

    def start(self):
        """Start the necessary channels to coordinate with runtime and group
        of processes this RuntimeService is managing"""
        self.runtime_to_service.start()
        self.service_to_runtime.start()
        for i in range(len(self.service_to_process)):
            self.service_to_process[i].start()
            self.process_to_service[i].start()
        self.run()

    @abstractmethod
    def run(self):
        """Override this method to implement the runtime service. The run
        method is invoked upon start which called when the execution is
        started by the runtime."""
        pass

    def join(self):
        """Stop the necessary channels to coordinate with runtime and group
        of processes this RuntimeService is managing"""
        self.runtime_to_service.join()
        self.service_to_runtime.join()

        for i in range(len(self.service_to_process)):
            self.service_to_process[i].join()
            self.process_to_service[i].join()


class CRuntimeService(AbstractRuntimeService):
    pass


class NcRuntimeService(AbstractRuntimeService):
    pass


class LoihiPyRuntimeService(PyRuntimeService):
    """RuntimeService that implements Loihi SyncProtocol in Python."""

    def __init__(self, protocol):
        super().__init__(protocol)
        self.req_pre_lrn_mgmt = False
        self.req_post_lrn_mgmt = False
        self.req_lrn = False
        self.req_stop = False
        self.req_pause = False
        self.paused = False
        self._error = False
        self.pausing = False
        self.stopping = False

    class Phase:
        SPK = enum_to_np(1)
        PRE_MGMT = enum_to_np(2)
        LRN = enum_to_np(3)
        POST_MGMT = enum_to_np(4)
        HOST = enum_to_np(5)

    class PMResponse:
        STATUS_DONE = enum_to_np(0)
        """Signfies Ack or Finished with the Command"""
        STATUS_TERMINATED = enum_to_np(-1)
        """Signifies Termination"""
        STATUS_ERROR = enum_to_np(-2)
        """Signifies Error raised"""
        STATUS_PAUSED = enum_to_np(-3)
        """Signifies Execution State to be Paused"""
        REQ_PRE_LRN_MGMT = enum_to_np(-4)
        """Signifies Request of PREMPTION"""
        REQ_LEARNING = enum_to_np(-5)
        """Signifies Request of LEARNING"""
        REQ_POST_LRN_MGMT = enum_to_np(-6)
        """Signifies Request of PREMPTION"""
        REQ_PAUSE = enum_to_np(-7)
        """Signifies Request of PAUSE"""
        REQ_STOP = enum_to_np(-8)
        """Signifies Request of STOP"""

    def _next_phase(self, is_last_time_step: bool):
        """Advances the current phase to the next phase.
        On the first time step it starts with HOST phase and advances to SPK.
        Afterwards it loops: SPK -> PRE_MGMT -> LRN -> POST_MGMT -> SPK
        On the last time step POST_MGMT advances to HOST phase."""
        if self.req_pre_lrn_mgmt:
            self.req_pre_lrn_mgmt = False
            return LoihiPyRuntimeService.Phase.PRE_MGMT
        if self.req_post_lrn_mgmt:
            self.req_post_lrn_mgmt = False
            return LoihiPyRuntimeService.Phase.POST_MGMT
        if self.req_lrn:
            self.req_lrn = False
            return LoihiPyRuntimeService.Phase.LRN
        if self.req_pause:
            self.req_pause = False
            return MGMT_COMMAND.PAUSE
        if self.req_stop:
            self.req_stop = False
            return MGMT_COMMAND.STOP

        if is_last_time_step:
            return LoihiPyRuntimeService.Phase.HOST
        return LoihiPyRuntimeService.Phase.SPK

    def _send_pm_cmd(self, phase: MGMT_COMMAND):
        """Sends a command (phase information) to all ProcessModels."""
        for send_port in self.service_to_process:
            send_port.send(phase)

    def _send_pm_req_given_model_id(self, model_id: int, *requests):
        """Sends requests to a ProcessModel given by the model id."""
        process_idx = self.model_ids.index(model_id)
        req_port = self.service_to_process[process_idx]
        for request in requests:
            req_port.send(request)

    def _get_pm_resp(self) -> ty.Iterable[MGMT_RESPONSE]:
        """Retrieves responses of all ProcessModels."""
        rcv_msgs = []
        num_responses_expected = len(self.model_ids)
        counter = 0
        while counter < num_responses_expected:
            ptos_recv_port = self.process_to_service[counter]
            rcv_msgs.append(ptos_recv_port.recv())
            counter += 1
        for idx, recv_msg in enumerate(rcv_msgs):
            if enum_equal(recv_msg,
                          LoihiPyRuntimeService.PMResponse.STATUS_ERROR):
                self._error = True
            if enum_equal(recv_msg,
                          LoihiPyRuntimeService.PMResponse.REQ_PRE_LRN_MGMT):
                self.req_pre_lrn_mgmt = True
            if enum_equal(recv_msg,
                          LoihiPyRuntimeService.PMResponse.REQ_POST_LRN_MGMT):
                self.req_post_lrn_mgmt = True
            if enum_equal(recv_msg,
                          LoihiPyRuntimeService.PMResponse.REQ_LEARNING):
                self.req_lrn = True
            if enum_equal(recv_msg,
                          LoihiPyRuntimeService.PMResponse.REQ_PAUSE):
                # ToDo: Add some mechanism to get the exact process id
                self.log.info(f"Process : {idx} has requested Pause")
                self.req_pause = True
            if enum_equal(recv_msg,
                          LoihiPyRuntimeService.PMResponse.REQ_STOP):
                # ToDo: Add some mechanism to get the exact process id
                self.log.info(f"Process : {idx} has requested Stop")
                self.req_stop = True
        return rcv_msgs

    def _relay_to_runtime_data_given_model_id(self, model_id: int):
        """Relays data received from ProcessModel given by model id  to the
        runtime"""
        process_idx = self.model_ids.index(model_id)
        data_recv_port = self.process_to_service[process_idx]
        data_relay_port = self.service_to_runtime
        num_items = data_recv_port.recv()
        data_relay_port.send(num_items)
        for i in range(int(num_items[0])):
            value = data_recv_port.recv()
            data_relay_port.send(value)

    def _relay_to_pm_data_given_model_id(self, model_id: int) -> MGMT_RESPONSE:
        """Relays data received from the runtime to the ProcessModel given by
        the model id."""
        process_idx = self.model_ids.index(model_id)
        data_recv_port = self.runtime_to_service
        data_relay_port = self.service_to_process[process_idx]
        resp_port = self.process_to_service[process_idx]
        # Receive and relay number of items
        num_items = data_recv_port.recv()
        data_relay_port.send(num_items)
        # Receive and relay data1, data2, ...
        for i in range(int(num_items[0].item())):
            data_relay_port.send(data_recv_port.recv())
        rsp = resp_port.recv()
        return rsp

    def _relay_pm_ack_given_model_id(self, model_id: int):
        """Relays ack received from ProcessModel given by model id to the
        runtime."""
        process_idx = self.model_ids.index(model_id)

        ack_recv_port = self.process_to_service[process_idx]
        ack_relay_port = self.service_to_runtime
        ack_relay_port.send(ack_recv_port.recv())

    def _handle_pause(self):
        # Inform all ProcessModels about the PAUSE command
        self._send_pm_cmd(MGMT_COMMAND.PAUSE)
        rsps = self._get_pm_resp()
        for rsp in rsps:
            if not enum_equal(rsp,
                              LoihiPyRuntimeService.PMResponse.STATUS_PAUSED):
                raise ValueError(f"Wrong Response Received : {rsp}")
        # Inform the runtime about successful pausing
        self.service_to_runtime.send(MGMT_RESPONSE.PAUSED)

    def _handle_stop(self):
        # Inform all ProcessModels about the STOP command
        self._send_pm_cmd(MGMT_COMMAND.STOP)
        rsps = self._get_pm_resp()
        for rsp in rsps:
            if not enum_equal(rsp,
                              LoihiPyRuntimeService.PMResponse.STATUS_TERMINATED
                              ):
                raise ValueError(f"Wrong Response Received : {rsp}")
        # Inform the runtime about successful termination
        self.service_to_runtime.send(MGMT_RESPONSE.TERMINATED)
        self.join()

    def run(self):
        """Retrieves commands from the runtime. On STOP or PAUSE commands all
        ProcessModels are notified and expected to TERMINATE or PAUSE,
        respectively. Otherwise the number of time steps is received as command.
        In this case iterate through the phases of the Loihi protocol until the
        last time step is reached. The runtime is informed after the last time
        step. The loop ends when receiving the STOP command from the runtime."""
        selector = CspSelector()
        phase = LoihiPhase.HOST

        channel_actions = [(self.runtime_to_service, lambda: 'cmd')]

        while True:
            # Probe if there is a new command from the runtime
            action = selector.select(*channel_actions)
            if action == 'cmd':
                command = self.runtime_to_service.recv()
                if enum_equal(command, MGMT_COMMAND.STOP):
                    self._handle_stop()
                    return
                elif enum_equal(command, MGMT_COMMAND.PAUSE):
                    self._handle_pause()
                    self.paused = True
                elif enum_equal(command, MGMT_COMMAND.GET_DATA) or \
                        enum_equal(command, MGMT_COMMAND.SET_DATA):
                    self._handle_get_set(phase, command)
                else:
                    self.paused = False
                    # The number of time steps was received ("command")
                    # Start iterating through Loihi phases
                    curr_time_step = 0
                    phase = LoihiPhase.HOST
                    is_last_ts = False
                    while True:
                        # Check if it is the last time step
                        is_last_ts = enum_equal(enum_to_np(curr_time_step),
                                                command)
                        # Advance to the next phase
                        phase = self._next_phase(is_last_ts)
                        if enum_equal(phase, MGMT_COMMAND.STOP):
                            if not self.stopping:
                                self.service_to_runtime.send(
                                    MGMT_RESPONSE.REQ_STOP)
                            phase = LoihiPhase.HOST
                            break
                        if enum_equal(phase, MGMT_COMMAND.PAUSE):
                            if not self.pausing:
                                self.service_to_runtime.send(
                                    MGMT_RESPONSE.REQ_PAUSE)
                            # Move to Host phase (get/set Var needs it)
                            phase = LoihiPhase.HOST
                            break
                        # Increase time step if spiking phase
                        if enum_equal(phase, LoihiPhase.SPK):
                            curr_time_step += 1
                        # Inform ProcessModels about current phase
                        self._send_pm_cmd(phase)
                        # ProcessModels respond with DONE if not HOST phase
                        if not enum_equal(
                                phase, LoihiPyRuntimeService.Phase.HOST):
                            self._get_pm_resp()
                            if self._error:
                                # Forward error to runtime
                                self.service_to_runtime.send(
                                    MGMT_RESPONSE.ERROR)
                                # stop all other pm
                                self._send_pm_cmd(MGMT_COMMAND.STOP)
                                return
                        # Check if pause or stop received from Runtime
                        if self.runtime_to_service.probe():
                            cmd = self.runtime_to_service.peek()
                            if enum_equal(cmd, MGMT_COMMAND.STOP):
                                self.stopping = True
                                self.req_stop = True
                            if enum_equal(cmd, MGMT_COMMAND.PAUSE):
                                self.pausing = True
                                self.req_pause = True

                        # If HOST phase (last time step ended) break the loop
                        if enum_equal(
                                phase, LoihiPhase.HOST):
                            break
                    if self.pausing or self.stopping or enum_equal(
                            phase, MGMT_COMMAND.STOP) or enum_equal(
                            phase, MGMT_COMMAND.PAUSE):
                        continue
                    # Inform the runtime that last time step was reached
                    if is_last_ts:
                        self.service_to_runtime.send(MGMT_RESPONSE.DONE)
            else:
                self.service_to_runtime.send(MGMT_RESPONSE.ERROR)

    def _handle_get_set(self, phase, command):
        if enum_equal(phase, LoihiPhase.HOST):
            if enum_equal(command, MGMT_COMMAND.GET_DATA):
                requests: ty.List[np.ndarray] = [command]
                # recv model_id
                model_id: int = int(self.runtime_to_service.recv()[0].item())
                # recv var_id
                requests.append(self.runtime_to_service.recv())
                self._send_pm_req_given_model_id(model_id, *requests)
                self._relay_to_runtime_data_given_model_id(model_id)
            elif enum_equal(command, MGMT_COMMAND.SET_DATA):
                requests: ty.List[np.ndarray] = [command]
                # recv model_id
                model_id: int = int(self.runtime_to_service.recv()[0].item())
                # recv var_id
                requests.append(self.runtime_to_service.recv())
                self._send_pm_req_given_model_id(model_id, *requests)
                rsp = self._relay_to_pm_data_given_model_id(model_id)
                self.service_to_runtime.send(rsp)
            else:
                raise RuntimeError(f"Unknown request {command}")


class LoihiCRuntimeService(AbstractRuntimeService):
    """RuntimeService that implements Loihi SyncProtocol in C."""
    pass


class AsyncPyRuntimeService(PyRuntimeService):
    """RuntimeService that implements Async SyncProtocol in Py."""

    def __init__(self, protocol):
        super().__init__(protocol)
        self.req_stop = False
        self.req_pause = False
        self._error = False

    class PMResponse:
        STATUS_DONE = enum_to_np(0)
        """Signfies Ack or Finished with the Command"""
        STATUS_TERMINATED = enum_to_np(-1)
        """Signifies Termination"""
        STATUS_ERROR = enum_to_np(-2)
        """Signifies Error raised"""
        STATUS_PAUSED = enum_to_np(-3)
        """Signifies Execution State to be Paused"""
        REQ_PAUSE = enum_to_np(-4)
        """Signifies Request of PAUSE"""
        REQ_STOP = enum_to_np(-5)
        """Signifies Request of STOP"""

    def _send_pm_cmd(self, cmd: MGMT_COMMAND):
        for stop_send_port in self.service_to_process:
            stop_send_port.send(cmd)

    def _get_pm_resp(self) -> ty.Iterable[MGMT_RESPONSE]:
        rcv_msgs = []
        for ptos_recv_port in self.process_to_service:
            rcv_msgs.append(ptos_recv_port.recv())
        return rcv_msgs

    def _handle_pause(self):
        # Inform the runtime about successful pausing
        self.service_to_runtime.send(MGMT_RESPONSE.PAUSED)

    def _handle_stop(self):
        self._send_pm_cmd(MGMT_COMMAND.STOP)
        rsps = self._get_pm_resp()
        for rsp in rsps:
            if not enum_equal(rsp,
                              LoihiPyRuntimeService.PMResponse.STATUS_TERMINATED
                              ):
                self.service_to_runtime.send(MGMT_RESPONSE.ERROR)
                raise ValueError(f"Wrong Response Received : {rsp}")
        # Inform the runtime about successful termination
        self.service_to_runtime.send(MGMT_RESPONSE.TERMINATED)
        self.join()

    def run(self):
        """Retrieves commands from the runtime and relays them to the process
        models. Also send the acknowledgement back to runtime."""
        selector = CspSelector()
        channel_actions = [(self.runtime_to_service, lambda: 'cmd')]
        while True:
            # Probe if there is a new command from the runtime
            action = selector.select(*channel_actions)
            channel_actions = []
            if action == 'cmd':
                command = self.runtime_to_service.recv()
                if enum_equal(command, MGMT_COMMAND.STOP):
                    self._handle_stop()
                    return
                elif enum_equal(command, MGMT_COMMAND.PAUSE):
                    self._handle_pause()
                else:
                    self._send_pm_cmd(MGMT_COMMAND.RUN)
                    for ptos_recv_port in self.process_to_service:
                        channel_actions.append((ptos_recv_port,
                                                lambda: 'resp'))
            elif action == 'resp':
                resps = self._get_pm_resp()
                for resp in resps:
                    if enum_equal(resp,
                                  AsyncPyRuntimeService.PMResponse.REQ_PAUSE):
                        self.req_pause = True
                    if enum_equal(resp,
                                  AsyncPyRuntimeService.PMResponse.REQ_STOP):
                        self.req_stop = True
                    if enum_equal(resp,
                                  AsyncPyRuntimeService.PMResponse.STATUS_ERROR
                                  ):
                        self._error = True
                if self.req_stop:
                    self.service_to_runtime.send(MGMT_RESPONSE.REQ_STOP)
                if self.req_pause:
                    self.service_to_runtime.send(MGMT_RESPONSE.REQ_PAUSE)
                if self._error:
                    self.service_to_runtime.send(MGMT_RESPONSE.ERROR)
            else:
                self.service_to_runtime.send(MGMT_RESPONSE.ERROR)
                raise ValueError(f"Wrong type of channel action : {action}")
            channel_actions.append((self.runtime_to_service, lambda: 'cmd'))


class NxSdkRuntimeService(NcRuntimeService):
    """The NxSdkRuntimeService uses NxCore to coordinate
    communication and executinon on Loihi within a SyncDomain.

    Parameters
    ----------
    protocol: ty.Type[AbstractSyncProtocol]
              Communication protocol used by NxSdkRuntimeService
    loihi_version: LoihiVersion
                   Version of Loihi Chip to use, N2 or N3
    loglevel: int
             Log level to use for logging
    """

    def __init__(self,
                 protocol: ty.Type[AbstractSyncProtocol],
                 loihi_version: LoihiVersion = LoihiVersion.N3,
                 loglevel: int = logging.WARNING):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(loglevel)
        super(NxSdkRuntimeService, self).__init__(
            protocol=protocol
        )
        self.board: NxBoard = None
        self.num_steps = 0

        try:
            if loihi_version == LoihiVersion.N3:
                from nxsdk.arch.n3b.n3board import N3Board
                # # TODO: Use dynamic set Board Init
                self.board = N3Board(1, 1, [2], [[5, 5]])
            elif loihi_version == LoihiVersion.N2:
                from nxsdk.arch.n2a.n2board import N2Board  # noqa F401
                self.board = N2Board(1, 1, [2], [[5, 5]])
            else:
                raise ValueError('Unsupported Loihi version '
                                 + 'used in board selection')
        except(ImportError):
            class NxBoard():
                pass
            self.board = NxBoard()

        self.log.debug("NxSdkRuntimeService is initialized")

    def run(self):
        """Retrieves commands from the runtime. STOP and PAUSE commands are
        relayed to NxCore. Otherwise the number of time steps is received as
        a RUN command. In this case RUN is relayed to NxCore with number of time
        steps. The loop ends when receiving the STOP command from the runtime.
        """

        self.log.debug("NxSdkRuntime is running")
        selector = CspSelector()
        channel_actions = [(self.runtime_to_service, lambda: 'cmd')]

        while True:
            action = selector.select(*channel_actions)
            if action == 'cmd':
                command = self.runtime_to_service.recv()
                self.log.debug("Recieved command: " + str(command[0]))
                if enum_equal(command, MGMT_COMMAND.STOP):
                    self.board.stop()

                    self.service_to_runtime.send(MGMT_RESPONSE.TERMINATED)
                    self.join()
                    return
                elif enum_equal(command, MGMT_COMMAND.PAUSE):
                    self.board.pause()

                    self.service_to_runtime.send(MGMT_RESPONSE.PAUSED)
                    break
                # If message recieved from Runtime is greater than zero
                # it is the num_steps for a run, use num_steps to start
                # a run
                elif command[0] > 0:
                    self.log.debug("Command: " + str(command[0])
                                   + " > 0, setting num_steps and running")
                    self.num_steps = command
                    self.board.run(numSteps=self.num_steps, aSync=False)

                    self.service_to_runtime.send(MGMT_RESPONSE.DONE)
                else:
                    self.service_to_runtime.send(MGMT_RESPONSE.ERROR)
                    return