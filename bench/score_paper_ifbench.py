"""Grade a complete IFBench run with the pinned official scorer."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess
import sys


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    output = args.directory
    dataset = args.checkout / "data/IFBench_test.jsonl"
    responses = output / "ifbench-responses.jsonl"
    protocol = json.loads((output / "ifbench-protocol.json").read_text())
    revision = subprocess.check_output(
        ["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != "270c25d8bf1bd6bfb10e69fdc4990432fc61f983":
        raise ValueError("Unexpected IFBench grader revision")
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != protocol["dataset_sha256"]:
        raise ValueError("Grading prompts differ from generation")
    questions, answers = rows(dataset), rows(responses)
    expected = [(q["key"], q["prompt"]) for q in questions]
    if len(set(expected)) != 300 or [(a["key"], a["prompt"]) for a in answers] != expected:
        raise ValueError("Need exactly one response per release prompt in dataset order")
    scores = output / "ifbench-scores"
    scores.mkdir(exist_ok=True)
    with (output / "ifbench-scoring.log").open("w") as log:
        subprocess.run(
            [sys.executable, str(args.checkout / "run_eval.py"), "--input_data", str(dataset),
             "--input_response_data", str(responses), "--output_dir", str(scores)],
            stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    result = {}
    for mode in ["strict", "loose"]:
        grades = rows(scores / f"ifbench-eval_results_{mode}.jsonl")
        passed_by_finish = Counter()
        instruction_pass = instruction_total = 0
        for answer, grade, q in zip(answers, grades, questions, strict=True):
            if (grade["prompt"] != q["prompt"] or grade["response"] != answer["response"]
                    or grade["instruction_id_list"] != q["instruction_id_list"]
                    or len(grade["follow_instruction_list"]) != len(q["instruction_id_list"])):
                raise ValueError(f"Official grader output does not match prompt {q['key']}")
            if grade["follow_all_instructions"]:
                passed_by_finish[answer["finish_reason"]] += 1
            instruction_pass += sum(grade["follow_instruction_list"])
            instruction_total += len(grade["follow_instruction_list"])
        passed = sum(passed_by_finish.values())
        result[mode] = dict(prompt_pass=passed, prompt_total=len(grades),
                            prompt_percent=100 * passed / len(grades),
                            instruction_pass=instruction_pass, instruction_total=instruction_total,
                            instruction_percent=100 * instruction_pass / instruction_total,
                            pass_by_finish_reason=passed_by_finish)
    result.update(
        dataset_sha256=protocol["dataset_sha256"], grader_revision=revision,
        sampled_completions=len(answers), sampling=protocol["sampling"],
        empty_final_answers=sum(not a["response"].strip() for a in answers),
        finish_reasons=Counter(a["finish_reason"] for a in answers),
        request_seconds=sum(a["seconds"] for a in answers),
        p50_seconds=median(a["seconds"] for a in answers),
        usage={k: sum(a["usage"][k] for a in answers)
               for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")},
    )
    (output / "ifbench-summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
