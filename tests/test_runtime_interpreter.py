import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'bench'))
from probe import validation_command,PROJECT_PYTHON


class RuntimeInterpreterTests(unittest.TestCase):
    def test_owned_quality_script_rebinds_without_changing_plan_or_foreign_commands(self):
        original=['/tmp/older-env/bin/python','bench/eval_quality.py','--tag','control']
        effective=validation_command(original)
        self.assertEqual(effective,[str(PROJECT_PYTHON),*original[1:]])
        self.assertEqual(original[0],'/tmp/older-env/bin/python')
        with tempfile.TemporaryDirectory() as tmp:
            script=Path(tmp)/'foreign.py';script.write_text('pass\n')
            foreign=[original[0],str(script)]
            self.assertEqual(validation_command(foreign),foreign)
        self.assertEqual(validation_command(['bash','bench/run_serving_bench.sh','control']),
                         ['bash','bench/run_serving_bench.sh','control'])

    @unittest.skipUnless(Path('/usr/bin/python3').is_file(),'Requires a separate system interpreter')
    def test_probe_bootstraps_project_before_dry_run_without_gpu(self):
        command=json.loads(subprocess.check_output(['/usr/bin/python3',str(ROOT/'bench/probe.py'),
            '--dry-run','--output','/tmp/unused-runtime-interpreter'],cwd=ROOT,text=True))
        self.assertEqual(command[0],str(PROJECT_PYTHON))

    def test_runtime_shell_keeps_project_python_and_evaluator_importable(self):
        command="""source scripts/runtime-env.sh
python - <<'PY'
import sys,json,importlib.util
print(json.dumps({'prefix':sys.prefix,'sgl_eval_importable':importlib.util.find_spec('sgl_eval') is not None}))
PY
"""
        result=json.loads(subprocess.check_output(['bash','-c',command],cwd=ROOT,text=True))
        self.assertEqual(Path(result['prefix']).resolve(),(ROOT/'.venv').resolve())
        self.assertTrue(result['sgl_eval_importable'])


if __name__=='__main__':unittest.main()
