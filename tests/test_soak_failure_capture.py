import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
import soak
from api_smoke import VisionCheckError


class SoakFailureCaptureTests(unittest.TestCase):
    def test_observation_retains_errors_and_never_passes_short_quality_run(self):
        evidence={'response':{'choices':[{'message':{'content':'red square'}}]},
                  'image_sha256':'b'*64,'expected_objects_verified':False}
        def fail():raise VisionCheckError('Missing expected image object: blue circle',evidence)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'
            with (patch.object(sys,'argv',['soak','--seconds','.05','--image','unused',
                     '--continue-vision-errors','--output',str(output)]),
                  patch('soak.api',return_value={}),patch('soak.chat',return_value={}),
                  patch('soak.generate',return_value={}),patch('soak.cases',return_value=[('image_input',fail)]),
                  patch('soak.gpu_sample',return_value={'gpus':[{'used_mib':20000,'free_mib':2000,'utilization_percent':50}]})):
                self.assertEqual(soak.main(),1)
            result=json.loads((output/'soak.json').read_text())
            self.assertTrue(result['vision_quality_errors'])
            self.assertEqual(result['vision_quality_errors'][0]['evidence'],evidence)
            self.assertNotIn('failed_request',result)
            self.assertFalse(result['full_plan_soak_gate'])
            self.assertFalse(result['runtime_stability_7200s_gate'])

    def test_failed_image_is_retained_without_becoming_a_completed_case(self):
        evidence={'response':{'choices':[{'message':{'content':'red rectangle'}}]},
                  'image_sha256':'a'*64,'expected_objects_verified':False}
        def fail():
            raise VisionCheckError('Missing expected image object: red square',evidence)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'
            with (patch.object(sys,'argv',['soak','--seconds','60','--image','unused','--output',str(output)]),
                  patch('soak.api',return_value={}),patch('soak.chat',return_value={}),
                  patch('soak.generate',return_value={}),
                  patch('soak.cases',return_value=[('image_input',fail)]),
                  patch('soak.gpu_sample',return_value={'gpus':[{'used_mib':20000,'free_mib':2000,'utilization_percent':50}]})):
                self.assertEqual(soak.main(),1)
            result=json.loads((output/'soak.json').read_text())
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['failed_request']['name'],'image_input')
            self.assertEqual(result['failed_request']['evidence'],evidence)
            self.assertFalse(result['full_plan_soak_gate'])
            self.assertNotIn('image_input',[row['name'] for row in result['requests']])

    def test_monitor_failure_before_first_request_still_writes_failure(self):
        class ImmediateThread:
            def __init__(self,target,**kwargs):self.target=target
            def start(self):self.target()
            def join(self,**kwargs):pass
            def is_alive(self):return False
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'
            with (patch.object(sys,'argv',['soak','--seconds','60','--output',str(output)]),
                  patch('soak.api',return_value={}),
                  patch('soak.threading.Thread',ImmediateThread),
                  patch('soak.gpu_sample',side_effect=OSError('device telemetry unavailable'))):
                self.assertEqual(soak.main(),1)
            result=json.loads((output/'soak.json').read_text())
            self.assertEqual(result['status'],'failed')
            self.assertIsNone(result['failed_request']['name'])
            self.assertTrue(result['monitor_errors'])
            self.assertFalse(result['full_plan_soak_gate'])


if __name__=='__main__':
    unittest.main()
