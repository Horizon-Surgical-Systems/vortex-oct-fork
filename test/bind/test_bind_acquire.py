import datetime
import numpy as np
import unittest
import vortex


class TestAcquire(unittest.TestCase):
    def test_vector_option(self):
        a = vortex.acquire.VectorOption()
        # TODO: Test ass soon as Options is bound

    def test_Externalclocks(self):
        # InternalClock
        a = vortex.acquire.InternalClock(1234)
        self.assertEqual(repr(a), "InternalClock(samples_per_second=1234)")
        b = a.copy()
        self.assertFalse(a is b)
        self.assertEqual(repr(a), repr(b))
        a = vortex.acquire.InternalClock()
        b.samples_per_second = int(8 * 1e8)
        self.assertEqual(repr(a), repr(b))

    def test_ExternalClock(self):
        # ExternalClock
        a = vortex.acquire.ExternalClock()
        self.assertEqual(
            repr(a),
            "ExternalClock(coupling=Coupling.AC, dual=False, edge=ClockEdge.Rising, level_ratio=0.50)",
        )
        b = a.copy()
        self.assertFalse(a is b)
        self.assertEqual(repr(a), repr(b))
        b.coupling = vortex.acquire.alazar.Coupling.DC
        b.dual = True
        b.edge = vortex.acquire.alazar.ClockEdge.Falling
        b.level_ratio = 0.6
        self.assertEqual(
            repr(b),
            "ExternalClock(coupling=Coupling.DC, dual=True, edge=ClockEdge.Falling, level_ratio=0.60)",
        )

    def test_SingleExternalTrigger(self):
        # SingleExternalTrigger
        a = vortex.acquire.SingleExternalTrigger(
            300,
            0.05,
            84,
            coupling=vortex.acquire.alazar.Coupling.AC,
            slope=vortex.acquire.alazar.TriggerSlope.Negative,
        )
        self.assertEqual(
            repr(a),
            "SingleExternalTrigger(range_millivolts=300, level_ratio=0.05, delay_samples=84, slope=TriggerSlope.Negative, coupling=Coupling.AC)",
        )

    def test_DualExternalTrigger(self):
        # DualExternalTrigger
        b = vortex.acquire.DualExternalTrigger(
            delay_samples=56,
            initial_slope=vortex.acquire.alazar.TriggerSlope.Positive,
            level_ratios=[0.0, 0.0],
        )
        self.assertEqual(
            repr(b),
            "DualExternalTrigger(range_millivolts=2500, level_ratios=[0.0, 0.0], delay_samples=56, initial_slope=TriggerSlope.Positive, coupling=Coupling.DC)",
        )

    def test_Input(self):
        # Input
        a = vortex.acquire.Input(vortex.acquire.alazar.Channel.E)
        self.assertEqual(a.range_millivolts, 400)
        self.assertEqual(a.impedance_ohms, 50)
        self.assertEqual(a.bytes_per_sample, 2)
        self.assertEqual(a.coupling, vortex.acquire.alazar.Coupling.DC)
        self.assertEqual(a.channel, vortex.acquire.alazar.Channel.E)
        b = a.copy()
        self.assertFalse(a is b)
        b.impedance_ohms = 40
        self.assertEqual(
            repr(b),
            "Input(channel=Channel.E, range_millivolts=400, impedance_ohms=40, coupling=Coupling.DC, bytes_per_samples=2)",
        )

    def test_AuxIOTrigger(self):
        # AuxIOTriggerOut
        a = vortex.acquire.AuxIOTriggerOut()
        b = vortex.acquire.AuxIOTriggerOut()
        c = a.copy()
        self.assertEqual(b, c)
        self.assertFalse(b is c)
        self.assertEqual(repr(c), "AuxIOTriggerOut")

    def test_AuxIOClockOut(self):
        # AuxIOClockOut
        a = vortex.acquire.AuxIOClockOut()
        b = vortex.acquire.AuxIOClockOut()
        c = a.copy()
        self.assertEqual(repr(c), "AuxIOClockOut")
        self.assertEqual(b, c)

    def test_AuxIOPacerOut(self):
        # AuxIOPaceroOut
        a = vortex.acquire.AuxIOPacerOut()
        b = vortex.acquire.AuxIOPacerOut()
        c = a.copy()
        c.divider = 4
        self.assertEqual(repr(c), "AuxIOPacerOut(divider=4)")
        self.assertNotEqual(b, c)

    def test_OCTIgnoreBadClock(self):
        # OCTIgnoreBadClock
        a = vortex.acquire.OCTIgnoreBadClock()
        b = vortex.acquire.OCTIgnoreBadClock()
        c = a.copy()
        c.good_seconds = 4.3e-5
        c.bad_seconds = 6e6
        self.assertEqual(
            repr(c), "OCTIgnoreBadClock(good_seconds=4.3e-05, bad_seconds=6000000.0)"
        )
        self.assertNotEqual(b, c)

    def test_AlazarConfigDevice(self):
        # AlazarConfigDevice
        a = vortex.acquire.AlazarConfigDevice(4, 5)
        b = vortex.acquire.AlazarConfigDevice()
        c = a.copy()
        self.assertEqual(repr(a), "AlazarConfigDevice(system_index=4, board_index=5)")
        self.assertEqual(repr(b), "AlazarConfigDevice(system_index=1, board_index=1)")
        self.assertFalse(c is a)
        c.system_index = 6
        self.assertEqual(repr(c), "AlazarConfigDevice(system_index=6, board_index=5)")

    # Results from AlazarConfig. Here to avoid duplication in AlazarConfig and AlazarGPUConfig
    inner_default_repr = "(\n\
    shape = [1000, 1024, 0],\n\
    channels_per_sample = 0,\n\
    samples_per_record = 1024,\n\
    records_per_block = 1000,\n\
    device = AlazarConfigDevice(system_index=1, board_index=1),\n\
    clock = InternalClock(samples_per_second=800000000),\n\
    trigger = SingleExternalTrigger(range_millivolts=2500, level_ratio=0.09, delay_samples=80, slope=TriggerSlope.Positive, coupling=Coupling.DC),\n\
    inputs = VectorInput[],\n\
    options = VectorOption[],\n\
    acquire_timeout = datetime.timedelta(seconds=1),\n\
    stop_on_error = True,\n\
    bytes_per_multisample = 0,\n\
    channel_mask = Channel.???,\n\
    samples_per_second = 800000000,\n\
    samples_per_second_is_known = True,\n\
    recommended_minimum_records_per_block = 1,"

    inner_nondefault_repr = "(\n\
    shape = [500, 512, 1],\n\
    channels_per_sample = 1,\n\
    samples_per_record = 512,\n\
    records_per_block = 500,\n\
    device = AlazarConfigDevice(system_index=1, board_index=1),\n\
    clock = InternalClock(samples_per_second=500000000),\n\
    trigger = SingleExternalTrigger(range_millivolts=1000, level_ratio=0.09, delay_samples=80, slope=TriggerSlope.Negative, coupling=Coupling.DC),\n\
    inputs = VectorInput[Input(channel=Channel.H, range_millivolts=400, impedance_ohms=50, coupling=Coupling.DC, bytes_per_samples=2)],\n\
    options = VectorOption[AuxIOTriggerOut],\n\
    acquire_timeout = datetime.timedelta(seconds=2),\n\
    stop_on_error = False,\n\
    bytes_per_multisample = 2,\n\
    channel_mask = Channel.H,\n\
    samples_per_second = 500000000,\n\
    samples_per_second_is_known = True,\n\
    recommended_minimum_records_per_block = 1024,"

    def test_AlazarConfig(self):
        # AlazarConfig
        a = vortex.acquire.AlazarConfig()
        self.assertEqual(repr(a), "AlazarConfig" + self.inner_default_repr + "\n)")
        b = a.copy()
        self.assertFalse(b is a)
        self.assertEqual(repr(a), repr(b))
        b.inputs.append(vortex.acquire.Input(vortex.acquire.alazar.Channel.H))
        b.samples_per_record = 512
        b.records_per_block = 500
        b.clock = vortex.acquire.InternalClock(400000000)
        b.trigger = vortex.acquire.SingleExternalTrigger(
            range_millivolts=1000, slope=vortex.acquire.alazar.TriggerSlope.Negative
        )
        b.options.append(vortex.acquire.AuxIOTriggerOut())
        b.acquire_timeout = datetime.timedelta(seconds=2)
        b.stop_on_error = False
        b.samples_per_second = 500000000
        b.validate()
        self.assertEqual(repr(b), "AlazarConfig" + self.inner_nondefault_repr + "\n)")

    def test_AlazarAcquisition(self):
        # AlazarAcquisition # Can't quite test initializing and start, since we don't know what board is in the running device
        a = vortex.acquire.AlazarConfig()
        c = vortex.acquire.AlazarAcquisition()
        self.assertEqual(repr(a), repr(c.config))
        self.assertEqual(c.running, False)

    def test_AlazarGPUConfig(self):
        # AlazarGPUConfig
        a = vortex.acquire.AlazarGPUConfig()
        self.assertEqual(
            repr(a),
            "AlazarGPUConfig" + self.inner_default_repr + "\n    gpu_device = 0,\n)",
        )
        b = a.copy()
        self.assertFalse(b is a)
        self.assertEqual(repr(a), repr(b))
        b.inputs.append(vortex.acquire.Input(vortex.acquire.alazar.Channel.H))
        b.samples_per_record = 512
        b.records_per_block = 500
        b.clock = vortex.acquire.InternalClock(400000000)
        b.trigger = vortex.acquire.SingleExternalTrigger(
            range_millivolts=1000, slope=vortex.acquire.alazar.TriggerSlope.Negative
        )
        b.options.append(vortex.acquire.AuxIOTriggerOut())
        b.acquire_timeout = datetime.timedelta(seconds=2)
        b.stop_on_error = False
        b.samples_per_second = 500000000
        b.validate()
        self.assertEqual(
            repr(b),
            "AlazarGPUConfig" + self.inner_nondefault_repr + "\n    gpu_device = 0,\n)",
        )

    def test_AlazarGPUAcquisition(self):
        # AlazarGPUAcquisition # Can't quite test initializing and start, since we don't know what board is in the running device
        a = vortex.acquire.AlazarGPUConfig()
        c = vortex.acquire.AlazarGPUAcquisition()
        self.assertEqual(repr(a), repr(c.config))
        self.assertEqual(c.running, False)


if __name__ == "__main__":
    unittest.main()
