"""Guard expensive recovery against retrying failures or stitching old duration."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SerialRecoveryTests(unittest.TestCase):
    def exercise(self, results):
        with tempfile.TemporaryDirectory() as temporary:
            calls = []
            outcomes = iter(results)

            def probe(tag, flags, steps):
                calls.append((tag, flags, steps))
                return next(outcomes)

            namespace = {'root': Path(temporary), 'py': '/project/python', 'json': json,
                         'state': {}, 'save': lambda: None, 'probe': probe}
            source = ROOT / 'results/driver-sources/serial-recovery-20261111.py'
            tree = ast.parse(source.read_text())
            names = {'token_step', 'api_step', 'retrieval_step', 'finish_runtime', 'runtime_observation'}
            module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                                      and n.name in names], type_ignores=[])
            exec(compile(module, str(source), 'exec'), namespace)
            namespace['runtime_observation']()
            return calls, namespace['state']

    def test_capacity_pass_with_validation_failure_does_not_retry_to_hide_failure(self):
        calls, state = self.exercise([
            {'status': 'failed', 'capacity_gate_passed': True},
            {'status': 'passed', 'capacity_gate_passed': True}])
        self.assertEqual([c[0] for c in calls], [
            'vision-runtime-observation-r2-088', 'vision-runtime-restart-r2-088'])
        observation = state['runtime_observation']
        self.assertFalse(observation['runtime_7200s_validated'])
        self.assertFalse(observation['full_mixed_7200s_validated'])

    def test_capacity_failure_selects_new_budget_and_full_uninterrupted_duration(self):
        calls, state = self.exercise([
            {'status': 'failed', 'capacity_gate_passed': False},
            {'status': 'passed', 'capacity_gate_passed': True},
            {'status': 'passed', 'capacity_gate_passed': True}])
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[1][0], 'vision-runtime-observation-r2-092-pin4')
        self.assertIn('0.92', calls[1][1])
        for tag, _, steps in calls[:2]:
            soak = [s for s in steps if 'bench/soak.py' in s['command']]
            self.assertEqual(len(soak), 1)
            cmd = soak[0]['command']
            self.assertEqual(cmd[cmd.index('--seconds') + 1], '7200')
            self.assertIn('--continue-vision-errors', cmd)
            self.assertEqual(cmd[cmd.index('--output') + 1], 'results/soak/' + tag)
        self.assertFalse(state['runtime_observation']['runtime_7200s_validated'])


if __name__ == '__main__':
    unittest.main()
