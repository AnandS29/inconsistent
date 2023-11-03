import subprocess


def test_toy_experiments(tmp_path):
    subprocess.check_call(
        [
            "python",
            "-m",
            "inconsistent_preferences.toy_experiments",
            "--env=1d",
            "--batch_size=8",
            "--lr=1e-4",
            "--num_iterations=5",
            f"--out_dir={tmp_path}",
        ],
    )
