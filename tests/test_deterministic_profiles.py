"""Compare actual probe-built control/draft flags, including profile overrides."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from serve import effective_argument


class DeterministicProfilesTests(unittest.TestCase):
    def test_eager_controls_disable_both_phases_and_match_target_settings(self):
        configs={c['name']:c for c in json.loads((ROOT/'configs/experiments.json').read_text())['experiments']}
        names=['deterministic-control-triton-eager','deterministic-eagle-triton-eager','deterministic-ngram-triton-eager']
        commands=[]
        for name in names:
            c=configs[name]
            command=json.loads(subprocess.check_output([sys.executable,str(ROOT/'bench/probe.py'),
                '--profile',c['profile'],'--dry-run','--output','/tmp/unused-eager-diagnostic',
                '--',*c['flags']],cwd=ROOT,text=True))
            graph=json.loads(effective_argument(command,'--cuda-graph-config'))
            self.assertEqual(graph,{'prefill':{'backend':'disabled'},'decode':{'backend':'disabled'}})
            commands.append(command)
        for command,name in zip(commands[1:],names[1:]):
            self.assertEqual(configs[name]['control'],names[0])
            for flag in ('--model-path','--attention-backend','--linear-attn-prefill-backend',
                         '--mem-fraction-static','--chunked-prefill-size','--max-mamba-cache-size','--max-total-tokens'):
                self.assertEqual(effective_argument(command,flag),effective_argument(commands[0],flag),flag)

    def test_flashinfer_deterministic_alignment_fits_chunk(self):
        configs = json.loads((ROOT / 'configs/experiments.json').read_text())['experiments']
        selected = [c for c in configs if c['name'].startswith('deterministic-') and 'flashinfer' in c['name']]
        self.assertEqual(len(selected), 6)
        for config in selected:
            chunk = int(effective_argument(config['flags'], '--chunked-prefill-size'))
            tile = int(config['env'].get('SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE', '4096'))
            self.assertGreater(tile, 0)
            self.assertLessEqual(tile, chunk, config['name'])
            self.assertEqual(chunk % tile, 0, config['name'])
            if config.get('control'):
                control = next(c for c in configs if c['name'] == config['control'])
                self.assertEqual(config['env']['SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE'],
                                 control['env']['SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE'])

    def test_embedding_drafts_match_control_after_probe_overrides(self):
        configs = {x['name']: x for x in json.loads((ROOT / 'configs/experiments.json').read_text())['experiments']}
        name = 'deterministic-embedding-control-flashinfer'
        def build(config):
            env = dict(os.environ, MODEL_PATH=str(ROOT / 'weights/target-embedding-nvfp4'))
            env.pop('DRAFT_MODEL_PATH', None)
            return json.loads(subprocess.check_output([sys.executable, str(ROOT / 'bench/probe.py'),
                '--profile', config['profile'], '--dry-run', '--output', '/tmp/unused-deterministic-profile',
                '--', *config['flags']], cwd=ROOT, env=env, text=True))
        control = build(configs[name])
        self.assertIsNone(effective_argument(control, '--speculative-algorithm'))
        for algorithm in ('dspark', 'dflash'):
            candidate = configs[f'deterministic-embedding-{algorithm}-flashinfer']
            command = build(candidate)
            self.assertEqual(candidate['control'], name)
            for flag in ('--model-path', '--attention-backend', '--linear-attn-prefill-backend',
                         '--mem-fraction-static', '--chunked-prefill-size', '--max-mamba-cache-size', '--max-total-tokens'):
                self.assertEqual(effective_argument(command, flag), effective_argument(control, flag), flag)
            self.assertEqual(effective_argument(command, '--max-mamba-cache-size'), '8')
            self.assertEqual(effective_argument(command, '--max-total-tokens'), '24576')
            self.assertEqual(effective_argument(command, '--speculative-algorithm'), algorithm.upper())
            for flag in ('--enable-deterministic-inference', '--disable-radix-cache'):
                self.assertIn(flag, command); self.assertIn(flag, control)
            self.assertEqual(candidate['env']['SGLANG_OPT_GDN_VERIFY_MATCH_PACKED'], '1')
            self.assertEqual(candidate['env']['SGLANG_OPT_GDN_VERIFY_CONV_ORDER'], '1')
            self.assertNotIn('SGLANG_OPT_QWEN_MTP_EARLY_SHARE', candidate['env'])


if __name__ == '__main__':
    unittest.main()
