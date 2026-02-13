import numpy as np
import unittest
import vortex

class TestEngine(unittest.TestCase):
    def test_find_rising_edges(self):
        # find_rising_edges
        self.assertTrue(np.allclose(vortex.engine.find_rising_edges(np.array([1, 2, 1, 2, 1, 2, 1]), 1), np.array([1.42857143, 3.42857143, 5.42857143])))
        self.assertTrue(np.allclose(vortex.engine.find_rising_edges(np.array([0, 1, 2, 3, 4, 3, 2, 1, 2, 3, 4]), 3), np.array([1.09090909, 3.09090909])))
        self.assertTrue(np.allclose(vortex.engine.find_rising_edges(np.array([0, 1, 2, 3, 4, 3, 2, 1, 2, 3, 4]), 3, 1), np.array([1.09090909])))
        self.assertTrue(np.allclose(vortex.engine.find_rising_edges(np.array([0, 1.23, 2, 3, 4, 3, 2, 1, 2, 3, 4]), 3), np.array([1.09787879, 3.09787879])))
        self.assertTrue(np.allclose(vortex.engine.find_rising_edges(np.array([0, 1.23, 2, 3, 4, 3, 2, 1, 2, 3, 4]), 3, 1), np.array([1.09787879])))

    def test_BroctStorageEndpoint(self):
        a = vortex.format.BroctFormatExecutor()
        b = vortex.storage.BroctStorage()
        c = vortex.get_python_logger("Debug")
        d = vortex.engine.BroctStorageEndpoint(a, b, c)
        d.allocate()
        self.assertEqual(d.storage, b)
        self.assertEqual(d.executor, a)

if __name__ == '__main__':
    unittest.main()
    
