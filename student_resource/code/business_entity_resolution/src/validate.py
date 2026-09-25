"""Wrapper around the official validate_submission.py script."""
import subprocess, sys, os
from .config import cfg, STUDENT_ROOT


def validate_submission(matching_path=None, candidate_path=None, test_dir=None, check_ids=False):
    """Run the official validator and return (exit_code, stdout)."""
    validator = STUDENT_ROOT / 'utils' / 'validate_submission.py'
    if not validator.exists():
        print(f"WARNING: Validator not found at {validator}")
        return -1, "Validator not found"
    
    matching_path = matching_path or os.path.join(cfg['paths']['output_dir'], 'matching_results.tsv')
    candidate_path = candidate_path or os.path.join(cfg['paths']['output_dir'], 'candidate_pairs.tsv')
    test_dir = test_dir or cfg['paths']['test_dir']
    
    cmd = [
        sys.executable, str(validator),
        '--matching', matching_path,
        '--candidate', candidate_path,
        '--test-dir', test_dir,
    ]
    if check_ids:
        cmd.append('--check-ids')
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(STUDENT_ROOT))
    output = result.stdout + result.stderr
    print(output)
    return result.returncode, output
