import numpy as np
import unittest
import vortex

class TestFormat(unittest.TestCase):
    def test_FormatPlannerConfig(self):
        a = vortex.format.FormatPlannerConfig()
        self.assertEqual(repr(a), "FormatPlannerConfig(shape=[0, 0], segments_per_volume=0, records_per_segments=0, stream_delay_sample=0, segment_mask=Flags(value=0xffffffffffffffff), strip_inactive=True, adapt_shape=True)")
        b = a.copy()
        self.assertEqual(repr(a), repr(b))

    def test_Copy(self):
        a = vortex.format.Copy()
        self.assertEqual(repr(a), "Copy(count=0, block_offset=0, buffer_segment=0, buffer_record=0, reverse=False)")
        a.count = 3
        a.block_offset=5
        a.buffer_segment=4
        a.buffer_record=5
        a.reverse=True
        self.assertEqual(repr(a), "Copy(count=3, block_offset=5, buffer_segment=4, buffer_record=5, reverse=True)")

    def test_Resize(self):
        a = vortex.format.Resize()
        a.shape = np.array([2, 3])
        self.assertEqual(repr(a), "Resize(segments_per_volume=2, records_per_segment=3)")

    def test_FinishSegment(self):
        a = vortex.format.FinishSegment()
        a.scan_index = 10
        a.volume_index = 4
        a.segment_index_buffer = 1
        self.assertEqual(repr(a), "FinishSegment(scan_index=10, volume_index=4, segment_index_buffer=1)")

    def test_FinishVolume(self):
        a = vortex.format.FinishVolume()
        a.scan_index = 5
        a.volume_index = 10
        self.assertEqual(repr(a), "FinishVolume(scan_index=5, volume_index=10)")

    def test_FinishScan(self):
        a = vortex.format.FinishScan()
        a.scan_index = 5
        self.assertEqual(repr(a), "FinishScan(scan_index=5)")

    def test_FormatPlanner(self):
        a = vortex.format.FormatPlanner()
        # TODO: test methods

    def test_FormatPlan(self):
        a = vortex.format.FormatPlan()
        a.append(vortex.format.Copy())
        a.append(vortex.format.Resize())
        a.append(vortex.format.FinishSegment())
        a.append(vortex.format.FinishVolume())
        a.append(vortex.format.FinishScan())
        self.assertEqual(repr(a), "FormatPlan[Copy(count=0, block_offset=0, buffer_segment=0, buffer_record=0, reverse=False), Resize(segments_per_volume=0, records_per_segment=0), FinishSegment(scan_index=0, volume_index=0, segment_index_buffer=0), FinishVolume(scan_index=0, volume_index=0), FinishScan(scan_index=0)]")

    def test_NullSlice(self):
        a = vortex.format.NullSlice()
        b = a.copy()
        self.assertEqual(repr(b), "NullSlice")

    def test_SimpleSlice(self):
        a = vortex.format.SimpleSlice()
        b = a.copy()
        b.start = 10
        b.stop = 100
        b.step = 5
        self.assertEqual(repr(b), "SimpleSlice(start=10, stop=100, step=5)")
        self.assertEqual(b.count(), 18)

    def test_NullTransform(self):
        a = vortex.format.NullTransform()
        b = a.copy()
        self.assertEqual(repr(b), "NullTransform")

    def test_LinearTransform(self):
        a = vortex.format.LinearTransform()
        b = a.copy()
        b.scale = 4.0
        b.offset = 34.5
        self.assertEqual(repr(b), "LinearTransform(scale=4.0, offset=34.5)")

    def test_StackFormatExecutorConfig(self):
        a = vortex.format.StackFormatExecutorConfig()
        b = a.copy()
        b.erase_after_volume = True
        b.sample_slice = vortex.format.SimpleSlice(0 , 10, 1)
        b.sample_transform = vortex.format.LinearTransform()
        self.assertEqual(repr(b), "StackFormatExecutorConfig(erase_after_volume=True, sample_slice=SimpleSlice(start=0, stop=10, step=1), sample_transform=LinearTransform(scale=1.0, offset=0.0))")
        
    def test_StackFormatExecutor(self):
        a = vortex.format.StackFormatExecutor()
        # TODO: bind methods

    def test_BroctFormatExecutor(self):
        a = vortex.format.BroctFormatExecutor()
        # TODO: bind methods

    def test_RadialFormatExecutorConfig(self):
        a = vortex.format.RadialFormatExecutorConfig()
        b = a.copy()
        b.erase_after_volume = True
        b.sample_slice = vortex.format.SimpleSlice(0 , 10, 1)
        b.sample_transform = vortex.format.LinearTransform()
        b.x_extent = vortex.Range(-10, 10)
        b.y_extent = vortex.Range(-10, 10)
        b.radial_extent = vortex.Range(-10, 10)
        b.angular_extent = vortex.Range(-10, 10)
        b.radial_shape = [1, 2]
        b.radial_segments_per_volume = 4
        b.radial_records_per_segment = 4
        self.assertEqual(repr(b), "RadialFormatExecutorConfig(\n\
    erase_after_volume=True,\n\
    sample_slice=SimpleSlice(start=0, stop=10, step=1),\n\
    sample_transform=LinearTransform(scale=1.0, offset=0.0),\n\
    volume_xy_extent=[Range(-10.0, 10.0), Range(-10.0, 10.0)],\n\
    x_extent=Range(-10.0, 10.0),\n\
    y_extent=Range(-10.0, 10.0),\n\
    segment_rt_extent=[Range(-10.0, 10.0), Range(-10.0, 10.0)],\n\
    radial_extent=Range(-10.0, 10.0),\n\
    angular_extent=Range(-10.0, 10.0),\n\
    radial_shape=[4, 4],\n\
    radial_segments_per_volume=4,\n\
    radial_records_per_segment=4\n)")
        
            
        
        
    def test_RadialFormatExecutor(self):
        a = vortex.format.RadialFormatExecutor()
        # TODO: bind methods
