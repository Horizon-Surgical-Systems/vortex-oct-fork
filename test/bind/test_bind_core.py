import unittest
import vortex

class TestBindCore(unittest.TestCase):
    Range_obj = vortex.Range()

    def test_non_default_constructor(self):
        Range_obj = vortex.Range(float('-inf'), int(0))

    def test_equal_operator(self):
        Range_obj = vortex.Range()
        self.assertEqual(Range_obj, vortex.Range())

    def test_properties(self):
        a = vortex.Range(float(-7.4), int(100))
        self.assertEqual(a.length, 107.4)
        self.assertEqual(a.min, -7.4)
        self.assertEqual(a.max, 100)

    def test_functions(self):
        a = vortex.Range.symmetric(1e6)
        self.assertEqual(a.min, -1e6)
        self.assertEqual(a.max, 1e6)
        self.assertTrue(a.contains(1e5))
        self.assertEqual(repr(a), "Range(-1000000.0, 1000000.0)")

    def test_logger_getters(self):
        vortex.get_console_logger("Console Logger")
        vortex.get_python_logger("Python Logger")
        

if __name__ == '__main__':
    unittest.main()
        
