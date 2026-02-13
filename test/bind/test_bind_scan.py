import numpy as np
import vortex
import unittest


class TestScan(unittest.TestCase):
    def test_Segment2D(self):
        a = vortex.scan.Segment2D()
        self.assertEqual(repr(a.position), "array([], shape=(0, 0), dtype=float64)")
        a.position = np.array([[1, 2], [5, 6], [3, 4]])
        a.markers.append(vortex.block.ActiveLines())
        a.entry_velocity = np.array([3.4, 4])
        a.exit_velocity = (0, 1)
        a.position[0][0] = 0
        self.assertEqual(
            repr(a),
            "Segment2D:\n\
array([[0., 2.],\n\
       [5., 6.],\n\
       [3., 4.]]),\n\
entry_velocity=(3.4, 4.0), exit_velocity=(0.0, 1.0), markers=DefaultMarkerVector[ActiveLines(sample=0)]\n",
        )

        b = a.copy()
        self.assertEqual(repr(a), repr(b))
        self.assertEqual(a.entry_position, (0.0, 2.0))
        self.assertEqual(a.exit_position, (3.0, 4.0))
        with self.assertRaises(TypeError):
            a.entry_position[0] = 2

    def test_Segment3D(self):
        a = vortex.scan.Segment3D()
        self.assertEqual(repr(a.position), "array([], shape=(0, 0), dtype=float64)")
        a.position = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        a.entry_velocity = (3, 4, 5)
        a.exit_velocity = np.array([1, 2, 0])
        a.markers.append(vortex.block.SegmentBoundary())
        a.position[0][0] = 0
        self.assertEqual(
            repr(a),
            "Segment3D:\n\
array([[0., 2., 3.],\n\
       [3., 4., 5.],\n\
       [5., 6., 7.]]),\n\
entry_velocity=(3.0, 4.0, 5.0), exit_velocity=(1.0, 2.0, 0.0), markers=DefaultMarkerVector[SegmentBoundary(sample=0, sequence=0, index_in_volume=0, reversed=False, record_count_hint=0, flags=Flags(value=0xffffffffffffffff))]\n",
        )
        b = a.copy()
        self.assertEqual(repr(a), repr(b))
        self.assertEqual(a.entry_position, (0.0, 2.0, 3.0))
        self.assertEqual(a.exit_position, (5.0, 6.0, 7.0))
        with self.assertRaises(TypeError):
            a.entry_position[0] = 2
