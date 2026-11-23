import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from report_soak import oom_events


class SoakLogEvidenceTests(unittest.TestCase):
    def test_loader_warning_is_not_an_observed_oom(self):
        warning="Triton kernel device-loaded after serving started. Pre-load it during engine init to avoid CUDA OOM."
        self.assertEqual(oom_events(warning),[])

    def test_recovered_and_fatal_events_remain_visible(self):
        text="INFO READY\nWARNING CUDA OOM encountered; retrying a smaller chunk\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate64MiB\n"
        events=oom_events(text)
        self.assertEqual([event['line'] for event in events],[2,3])


if __name__=='__main__':unittest.main()
