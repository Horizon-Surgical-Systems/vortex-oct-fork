import numpy as np
import unittest
import vortex

class TestBlock(unittest.TestCase):
    def test_Base(self):
        # Base
        a = vortex.block.Base()
        self.assertEqual(repr(a), "Base(sample=0)")
        a.sample = 5
        self.assertEqual(repr(a), "Base(sample=5)")
        b = vortex.block.Base(sample=6)
        self.assertEqual(b.sample, 6)

    def test_ScanBoundary(self):
        # ScanBoundary
        a = vortex.block.ScanBoundary(sample=6, sequence=10, volume_count_hint=2)
        self.assertEqual(repr(a), "ScanBoundary(sample=6, sequence=10, volume_count_hint=2)")
        a = vortex.block.ScanBoundary(9, 4)
        self.assertEqual(a.sample, 9)
        self.assertEqual(a.sequence, 4)
        self.assertEqual(a.volume_count_hint, 0)

    def test_VolumeBoundary(self):
        # VolumeBoundary
        a = vortex.block.VolumeBoundary()
        self.assertEqual(repr(a), "VolumeBoundary(sample=0, sequence=0, index_in_scan=0, reversed=False, segment_count_hint=0)")
        a = vortex.block.VolumeBoundary(10, 3, 4, True)
        self.assertEqual(a.sample, 10)
        self.assertEqual(a.sequence, 3)
        self.assertEqual(a.index_in_scan, 4)
        self.assertEqual(a.reversed, True)
        self.assertEqual(a.segment_count_hint, 0)

    def test_Flags(self):
        # Flags
        a = vortex.block.Flags()
        self.assertEqual(repr(a), "Flags(value=0x0)")
        a.value = 40
        b = a.copy()
        self.assertEqual(repr(b), "Flags(value=0x28)")
        self.assertEqual(a, b)
        c = vortex.block.Flags.all()
        b.set()
        self.assertEqual(c, b)
        a.value = 0
        c.clear()
        d = vortex.block.Flags.none()
        self.assertEqual(a, c)
        self.assertEqual(a, d)
        b.value = 7
        b.set(7)
        self.assertEqual(repr(b), "Flags(value=0x87)")

    def test_SegmentBoundary(self):
        # SegmentBoundary
        a = vortex.block.SegmentBoundary()
        a.sample = 6
        a.sequence = 10
        a.index_in_volume = 3
        a.reversed = False
        a.record_count_hint = 2
        b = vortex.block.SegmentBoundary(6, 10, 3, False, 2)
        d = vortex.block.Flags.none()
        b.flags = d
        a.flags = b.flags
        self.assertEqual(repr(a), repr(b))
        self.assertEqual(repr(a), "SegmentBoundary(sample=6, sequence=10, index_in_volume=3, reversed=False, record_count_hint=2, flags=Flags(value=0x0))")

    def test_ActiveLines(self):
        # ActiveLines
        a = vortex.block.ActiveLines()
        self.assertEqual(repr(a), "ActiveLines(sample=0)")

    def test_InactiveLines(self):
        # InactiveLines
        b = vortex.block.InactiveLines()
        self.assertEqual(repr(b), "InactiveLines(sample=0)")

    def test_ParameterChange(self):
        # ParameterChange
        c = vortex.block.ParameterChange()
        self.assertEqual(repr(c), "ParameterChange(sample=0, uuid=0)")
        c.uuid = 6
        self.assertEqual(repr(c), "ParameterChange(sample=0, uuid=6)")

    def test_DefaultMarkerVector(self):
        # DefaultMarkerVector
        a = vortex.block.ActiveLines()
        b = vortex.block.InactiveLines()
        c = vortex.block.ParameterChange()
        c.uuid = 6
        d = vortex.block.DefaultMarkerVector()
        seg_bound = vortex.block.SegmentBoundary()
        scan_bound = vortex.block.ScanBoundary()
        vol_boundary = vortex.block.VolumeBoundary()
        d.append(a)
        d.append(b)
        d.append(c)
        d.append(scan_bound)
        d.append(seg_bound)
        d.append(vol_boundary)
        self.assertEqual(repr(d), "DefaultMarkerVector[ActiveLines(sample=0), InactiveLines(sample=0), ParameterChange(sample=0, uuid=6), ScanBoundary(sample=0, sequence=0, volume_count_hint=0), SegmentBoundary(sample=0, sequence=0, index_in_volume=0, reversed=False, record_count_hint=0, flags=Flags(value=0xffffffffffffffff)), VolumeBoundary(sample=0, sequence=0, index_in_scan=0, reversed=False, segment_count_hint=0)]")

    def test_Int82D(self):
        # Int8, 2D
        a = vortex.block.StreamInt82D()
        b = a.copy()
        self.assertFalse(a is b)
        self.assertEqual(repr(a), repr(b))
        self.assertEqual(repr(a), "Stream:\narray([], shape=(0, 0), dtype=int8)")
        b.view = np.ones((3, 4000), dtype=np.int8)
        self.assertEqual(repr(b), "Stream:\n\
array([[1, 1, 1, ..., 1, 1, 1],\n\
       [1, 1, 1, ..., 1, 1, 1],\n\
       [1, 1, 1, ..., 1, 1, 1]], dtype=int8)")

    def test_UInt161D(self):
        # UInt16, 1D
        a = vortex.block.StreamUInt161D()
        b = a.copy()
        self.assertFalse(a is b)
        self.assertEqual(repr(a), repr(b))
        self.assertEqual(repr(a), "Stream:\narray([], dtype=uint16)")
        b.view = np.array([1, 2, 3, 4, 5, 6])
        self.assertEqual(repr(b), "Stream:\n\
array([1, 2, 3, 4, 5, 6], dtype=uint16)")
        
    def test_Int323D(self):
        # Int32, 3D
        a = vortex.block.StreamInt323D()
        b = a.copy()
        self.assertFalse(a is b)
        self.assertEqual(repr(a), repr(b))
        self.assertEqual(repr(a), "Stream:\narray([], shape=(0, 0, 0), dtype=int32)")
        b.view = np.zeros((3, 3, 3))
        self.assertEqual(repr(b), "Stream:\n\
array([[[0, 0, 0],\n\
        [0, 0, 0],\n\
        [0, 0, 0]],\n\
\n\
       [[0, 0, 0],\n\
        [0, 0, 0],\n\
        [0, 0, 0]],\n\
\n\
       [[0, 0, 0],\n\
        [0, 0, 0],\n\
        [0, 0, 0]]])")
        
    def test_Double2D(self):
        # Double, 2D
        a = vortex.block.StreamFloat642D()
        b = a.copy()
        self.assertFalse(a is b)
        self.assertEqual(repr(a), repr(b))
        self.assertEqual(repr(a), "Stream:\narray([], shape=(0, 0), dtype=float64)")
        b.view = np.ones((3, 5)) * 1.3
        self.assertEqual(repr(b), "Stream:\n\
array([[1.3, 1.3, 1.3, 1.3, 1.3],\n\
       [1.3, 1.3, 1.3, 1.3, 1.3],\n\
       [1.3, 1.3, 1.3, 1.3, 1.3]])")

    def test_Int8(self):
        # Int8
        a = vortex.block.CudaHostStreamInt8()
        self.assertEqual(repr(a), "CudaHostStream(shape=[], dtype='int8')")
        a.resize(np.array([1, 4]))
        self.assertEqual(repr(a), "CudaHostStream(shape=[1, 4], dtype='int8')")
        self.assertEqual(a.size_in_bytes, 4)
        self.assertEqual(a.dimension, 2)
        self.assertEqual(a.stride, [0, 1])
        self.assertEqual(a.resides_on_device, False)
        self.assertEqual(a.resides_on_host, True)
        
    def test_UInt16(self):
        # UInt16
        a = vortex.block.CudaHostStreamUInt16()
        self.assertEqual(repr(a), "CudaHostStream(shape=[], dtype='uint16')")
        a.resize(np.array([20, 20]))
        self.assertEqual(repr(a), "CudaHostStream(shape=[20, 20], dtype='uint16')")
        self.assertEqual(a.size_in_bytes, 800)
        self.assertEqual(a.dimension, 2)
        self.assertEqual(a.stride, [20, 1])
        self.assertEqual(a.resides_on_device, False)
        self.assertEqual(a.resides_on_host, True)

    def test_Int32(self):
        # Int32
        a = vortex.block.CudaHostStreamInt32()
        self.assertEqual(repr(a), "CudaHostStream(shape=[], dtype='int32')")
        a.resize(np.array([20, 20, 1000]))
        self.assertEqual(repr(a), "CudaHostStream(shape=[20, 20, 1000], dtype='int32')")
        self.assertEqual(a.size_in_bytes, 1600000)
        self.assertEqual(a.dimension, 3)
        self.assertEqual(a.stride, [20000, 1000, 1])
        self.assertEqual(a.resides_on_device, False)
        self.assertEqual(a.resides_on_host, True)

    def test_Double(self):
        # Double
        a = vortex.block.CudaHostStreamFloat64()
        self.assertEqual(repr(a), "CudaHostStream(shape=[], dtype='float64')")
        a.resize(np.array([20, 20, 1000]))
        self.assertEqual(repr(a), "CudaHostStream(shape=[20, 20, 1000], dtype='float64')")
        self.assertEqual(a.size_in_bytes, 3200000)
        self.assertEqual(a.dimension, 3)
        self.assertEqual(a.stride, [20000, 1000, 1])
        self.assertEqual(a.resides_on_device, False)
        self.assertEqual(a.resides_on_host, True)
        
        
    def test_cuda_Int8(self):
        # Int8
        a = vortex.block.CudaDeviceStreamInt8()
        self.assertEqual(repr(a), "CudaDeviceStream(shape=[], dtype='int8', device=-1)")
        a.resize(np.array([1, 4]))
        self.assertEqual(repr(a), "CudaDeviceStream(shape=[1, 4], dtype='int8', device=0)")
        self.assertEqual(a.size_in_bytes, 4)
        self.assertEqual(a.dimension, 2)
        self.assertEqual(a.stride, [0, 1])
        self.assertEqual(a.resides_on_host, False)
        self.assertEqual(a.resides_on_device, True)
        self.assertEqual(a.device, 0)
        
    def test_cuda_UInt16(self):
        # UInt16
        a = vortex.block.CudaDeviceStreamUInt16()
        self.assertEqual(repr(a), "CudaDeviceStream(shape=[], dtype='uint16', device=-1)")
        a.resize(np.array([20, 20]))
        self.assertEqual(repr(a), "CudaDeviceStream(shape=[20, 20], dtype='uint16', device=0)")
        self.assertEqual(a.size_in_bytes, 800)
        self.assertEqual(a.dimension, 2)
        self.assertEqual(a.stride, [20, 1])
        self.assertEqual(a.resides_on_host, False)
        self.assertEqual(a.resides_on_device, True)
        self.assertEqual(a.device, 0)

    def test_cuda_Int32(self):
        # Int32
        a = vortex.block.CudaDeviceStreamInt32()
        self.assertEqual(repr(a), "CudaDeviceStream(shape=[], dtype='int32', device=-1)")
        a.resize(np.array([20, 20, 1000]))
        self.assertEqual(repr(a), "CudaDeviceStream(shape=[20, 20, 1000], dtype='int32', device=0)")
        self.assertEqual(a.size_in_bytes, 1600000)
        self.assertEqual(a.dimension, 3)
        self.assertEqual(a.stride, [20000, 1000, 1])
        self.assertEqual(a.resides_on_host, False)
        self.assertEqual(a.resides_on_device, True)
        self.assertEqual(a.device, 0)

    def test_cuda_Double(self):
        # Double
        a = vortex.block.CudaDeviceStreamFloat64()
        self.assertEqual(repr(a), "CudaDeviceStream(shape=[], dtype='float64', device=-1)")
        a.resize(np.array([20, 20, 1000]))
        self.assertEqual(repr(a), "CudaDeviceStream(shape=[20, 20, 1000], dtype='float64', device=0)")
        self.assertEqual(a.size_in_bytes, 3200000)
        self.assertEqual(a.dimension, 3)
        self.assertEqual(a.stride, [20000, 1000, 1])
        self.assertEqual(a.resides_on_host, False)
        self.assertEqual(a.resides_on_device, True)
        self.assertEqual(a.device, 0)




if __name__ == '__main__':
    unittest.main()
    
