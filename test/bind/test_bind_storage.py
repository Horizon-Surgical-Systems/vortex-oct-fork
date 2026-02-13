import unittest
import vortex

class TestStorage(unittest.TestCase):
    def test_StreamDumpConfig(self):
        # StreamDumpConfig
        sconfig_obj = vortex.storage.StreamDumpConfig("my_path", 3, 45, True)
        self.assertEqual(repr(sconfig_obj), "StreamDumpConfig(path='my_path', stream=3, divisor=45, buffering=True)")
        s_copy_obj = sconfig_obj.copy()
        self.assertEqual(repr(sconfig_obj), repr(s_copy_obj))

    def test_StreamDump(self):
        # StreamDump
        sconfig_obj = vortex.storage.StreamDumpConfig("my_path", 3, 45, True)
        s_copy_obj = sconfig_obj.copy()
        sdump_obj = vortex.storage.StreamDump()
        sdump_obj.open(sconfig_obj)
        sdump_obj.config.stream = 5
        sdump_obj.config.buffering = False
        s_copy_obj.stream = 5
        s_copy_obj.buffering = False
        self.assertNotEqual(repr(sdump_obj.config), repr(s_copy_obj))
        self.assertEqual(repr(sdump_obj.config), repr(sconfig_obj))
        
    def test_BroctScan(self):
        # BroctScan
        bro_scan = vortex.storage.BroctScan(11)  # Edge case, if new types are added, this must be updated
        self.assertEqual(repr(bro_scan), "BroctScan.???")
        bro_scan = vortex.storage.BroctScan(10)
        self.assertEqual(repr(bro_scan), "BroctScan.spiral")
        bro_scan = vortex.storage.BroctScan.rectangular
        self.assertEqual(bro_scan, vortex.storage.BroctScan.rectangular)

    def test_BroctStorageConfig(self):
        # BroctStorageConfig
        bros_config_obj = vortex.storage.BroctStorageConfig("my_path", [1, 2, 3], [2, 2, 2], vortex.storage.BroctScan.spiral, buffering=True)
        self.assertEqual(repr(bros_config_obj), "BroctStorageConfig(path='my_path', shape=[1, 2, 3], dimensions=[2.0, 2.0, 2.0], scan_type=BroctScan.spiral, notes='', buffering=True)")
        self.assertEqual(bros_config_obj.samples_per_ascan, 3)
        self.assertEqual(bros_config_obj.ascans_per_bscan, 2)
        self.assertEqual(bros_config_obj.bscans_per_volume, 1)
        self.assertEqual(bros_config_obj.broct_volume_shape, [1, 3, 2])
        self.assertEqual(bros_config_obj.broct_bscan_shape, [3, 2])
        bros_c_copy = bros_config_obj.copy()
        self.assertEqual(repr(bros_c_copy), repr(bros_config_obj))

    def test_BroctStorage(self):
        # BroctStorage
        bros_config_obj = vortex.storage.BroctStorageConfig("my_path", [1, 2, 3], [2, 2, 2], vortex.storage.BroctScan.spiral, buffering=True)
        bstorage_obj = vortex.storage.BroctStorage()
        bstorage_obj.open(bros_config_obj)
        self.assertEqual(repr(bstorage_obj.config), repr(bros_config_obj))
                
    
if __name__ == '__main__':
    unittest.main()
    
