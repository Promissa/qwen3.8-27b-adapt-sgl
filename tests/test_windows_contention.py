"""A live native process alone is insufficient pressure/release evidence."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from windows_contention import native_active, native_released


class WindowsContentionTests(unittest.TestCase):
    def ready(self):
        return {'status':'active', 'vendor_id':0x10DE, 'windows_pid':123,
                'nominal_resource_bytes':512*1024**2, 'resource_count':8,
                'completed_copy_passes':1}

    def test_requires_nvidia_resources_and_completed_gpu_copy(self):
        self.assertTrue(native_active(self.ready(),512))
        for key, value in [('vendor_id',0x1002), ('nominal_resource_bytes',256*1024**2),
                           ('resource_count',4), ('completed_copy_passes',0),
                           ('windows_pid',0), ('status','starting')]:
            report=self.ready();report[key]=value
            self.assertFalse(native_active(report,512),key)

    def test_requires_successful_release_without_transferring_residency_claim(self):
        report=self.ready();report.update(status='completed',resources_released=True)
        self.assertTrue(native_released(report,512))
        for key, value in [('status','failed'), ('resources_released',False),
                           ('nominal_resource_bytes',0), ('completed_copy_passes',0)]:
            broken=deepcopy(report);broken[key]=value
            self.assertFalse(native_released(broken,512),key)


if __name__ == '__main__':
    unittest.main()
