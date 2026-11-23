"""Do not turn one-load gains, observed noise or failed controls into adoption."""
import ast
import copy
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/driver-sources/numerical-backend-followups-20261111.py'
tree = ast.parse(SOURCE.read_text())
namespace = {}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                             and n.name == 'select_quality'], type_ignores=[]), str(SOURCE), 'exec'), namespace)
select_quality = namespace['select_quality']


def summary(short, long):
    return {'status': 'passed', 'groups': [
        {'input_tokens': isl, 'metrics': {'output_throughput': {'values': values}}}
        for isl, values in [(1024, short), (8192, long)]]}


class NumericalSelectionTests(unittest.TestCase):
    def test_both_loads_and_observed_bracketing_control_range_required(self):
        before = summary([70, 71, 72], [64, 65, 66])
        after = summary([69, 70, 71], [63, 64, 65])
        candidate = summary([75, 76, 77], [68, 69, 70])
        self.assertTrue(select_quality(before, after, candidate)[0])
        one_load = summary([75, 76, 77], [63, 64, 65])
        self.assertFalse(select_quality(before, after, one_load)[0])
        # A large mean is insufficient when a replicate overlaps observed noise.
        noisy = summary([70, 80, 90], [68, 69, 70])
        self.assertFalse(select_quality(before, after, noisy)[0])
        incomplete = copy.deepcopy(before);incomplete['status'] = 'incomplete'
        with self.assertRaisesRegex(ValueError, 'complete'):
            select_quality(incomplete, after, candidate)


if __name__ == '__main__':
    unittest.main()
