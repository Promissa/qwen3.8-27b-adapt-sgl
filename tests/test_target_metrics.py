import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from target_metrics import summarize_events

def event(t,n):
 return {'at':t,'data':{'meta_info':{'completion_tokens':n,'prompt_tokens':8192}}}
class StreamMetricsTests(unittest.TestCase):
 def test_prefill_and_decode_are_separate_and_duplicate_terminal_is_not_a_token(self):
  r=summarize_events([event(1,1),event(2,3),event(2.1,3)],0,2.2,3)
  self.assertEqual(r['ttft_ms'],1000)
  self.assertEqual(r['decode_tok_s'],2)
  self.assertAlmostEqual(r['output_throughput'],3/2.2)
 def test_speculative_first_chunk_excludes_all_tokens_before_decode_window(self):
  r=summarize_events([event(1,4),event(2,8)],0,2.1,8)
  self.assertEqual(r['decode_window_tokens'],4)
  self.assertEqual(r['decode_tok_s'],4)
 def test_incomplete_count_and_one_chunk_cannot_prove_decode_speed(self):
  for es in [[event(1,2)],[event(1,3)],[event(1,3),event(2,2),event(3,3)]]:
   with self.assertRaises(ValueError):summarize_events(es,0,4,3)
if __name__=='__main__':unittest.main()
