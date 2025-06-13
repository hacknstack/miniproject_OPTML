import subprocess
import sys

def run_notebooks(notebooks, timeout=None, kernel=None):
    """
    Execute each notebook in the given list, in place.
    :param notebooks: list of .ipynb filenames
    :param timeout: max seconds per cell (None = no timeout)
    :param kernel: kernel name, e.g. 'python3'
    """
    base_cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace"
    ]
    if timeout is not None:
        base_cmd += [f"--ExecutePreprocessor.timeout={timeout}"]
    if kernel is not None:
        base_cmd += [f"--ExecutePreprocessor.kernel_name={kernel}"]

    for nb in notebooks:
        print(f"▶ Running {nb} …")
        proc = subprocess.run(base_cmd + [nb], capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"✖ Error in {nb}:\n{proc.stderr}")
            sys.exit(proc.returncode)
        print(f"✔ {nb} completed.\n")

if __name__ == "__main__":
    notebooks_to_run = [
        "feature_shift.ipynb",
        "label_shift.ipynb"
    ]
    # you can adjust timeout or kernel as needed:
    run_notebooks(notebooks_to_run, timeout=600, kernel="python3")