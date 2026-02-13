import datetime
import numpy as np
import unittest
import vortex

class TestDaqmx(unittest.TestCase):
    def test_Edge(self):
        # Edge
        a = vortex.io.daqmx.Edge(0)
        self.assertEqual(repr(a), "Edge.???")
        b = vortex.io.daqmx.Edge(10280)
        self.assertEqual(repr(b), "Edge.rising")
        c = vortex.io.daqmx.Edge(10171)
        self.assertEqual(repr(c), "Edge.falling")
        self.assertEqual(c, vortex.io.daqmx.Edge.falling)

    def test_Terminal(self):
        # Terminal
        a = vortex.io.daqmx.Terminal(0)
        self.assertEqual(repr(a), "Terminal.???")
        b = vortex.io.daqmx.Terminal(10083)
        self.assertEqual(repr(b), "Terminal.referenced")
        c = vortex.io.daqmx.Terminal(10106)
        self.assertEqual(repr(c), "Terminal.differential")
        self.assertEqual(c, vortex.io.daqmx.Terminal.differential)


class TestCuda(unittest.TestCase):
    def test_DeviceTensorFloat32(self):
        # DeviceTensor, Float32
        a = vortex.memory.DeviceTensorFloat32()
        self.assertTrue(a.on_device)
        self.assertFalse(a.on_host)
        a.resize([3, 4, 5])
        self.assertEqual(repr(a), "DeviceTensor(shape=[3, 4, 5], dtype='float32', device=0)")
        self.assertEqual(a.size_in_bytes, 240)
        self.assertEqual(a.count, 60)
        self.assertEqual(a.dimension, 3)
        self.assertEqual(a.stride, [20, 5, 1])

    def test_DeviceTensorInt8(self):
        # DeviceTensor, Int8
        a = vortex.memory.DeviceTensorInt8()
        self.assertTrue(a.on_device)
        self.assertFalse(a.on_host)
        a.resize([3, 4, 5])
        self.assertEqual(repr(a), "DeviceTensor(shape=[3, 4, 5], dtype='int8', device=0)")
        self.assertEqual(a.size_in_bytes, 60)
        self.assertEqual(a.count, 60)
        self.assertEqual(a.dimension, 3)
        self.assertEqual(a.stride, [20, 5, 1])

    def test_DeviceTensorUInt16(self):
        # DeviceTensor, UInt16
        a = vortex.memory.DeviceTensorUInt16()
        self.assertTrue(a.on_device)
        self.assertFalse(a.on_host)
        a.resize([3, 4, 5])
        self.assertEqual(repr(a), "DeviceTensor(shape=[3, 4, 5], dtype='uint16', device=0)")
        self.assertEqual(a.size_in_bytes, 120)
        self.assertEqual(a.count, 60)
        self.assertEqual(a.dimension, 3)
        self.assertEqual(a.stride, [20, 5, 1])

    def test_HostTensorFloat32(self):
        # HostTensor, Float32
        a = vortex.memory.HostTensorFloat32()
        self.assertTrue(a.on_host)
        self.assertFalse(a.on_device)
        a.resize([3, 4, 5])
        self.assertEqual(repr(a), "HostTensor(shape=[3, 4, 5], dtype='float32')")
        self.assertEqual(a.size_in_bytes, 240)
        self.assertEqual(a.count, 60)
        self.assertEqual(a.dimension, 3)
        self.assertEqual(a.stride, [20, 5, 1])

    def test_HostTensorInt8(self):
        # HostTensor, Int8
        a = vortex.memory.HostTensorInt8()
        self.assertTrue(a.on_host)
        self.assertFalse(a.on_device)
        a.resize([3, 4, 5])
        self.assertEqual(repr(a), "HostTensor(shape=[3, 4, 5], dtype='int8')")
        self.assertEqual(a.size_in_bytes, 60)
        self.assertEqual(a.count, 60)
        self.assertEqual(a.dimension, 3)
        self.assertEqual(a.stride, [20, 5, 1])

    def test_HostTensorUInt16(self):
        # HostTensorm UInt16
        a = vortex.memory.HostTensorUInt16()
        self.assertTrue(a.on_host)
        self.assertFalse(a.on_device)
        a.resize([3, 4, 5])
        self.assertEqual(repr(a), "HostTensor(shape=[3, 4, 5], dtype='uint16')")
        self.assertEqual(a.size_in_bytes, 120)
        self.assertEqual(a.count, 60)
        self.assertEqual(a.dimension, 3)
        self.assertEqual(a.stride, [20, 5, 1])
