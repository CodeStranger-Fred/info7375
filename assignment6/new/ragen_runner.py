"""
Modal runner for RAGEN with A*PO training on cloud GPUs.
"""

import modal
from pathlib import Path

image = (
    modal.Image.from_registry("pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime")
    .pip_install([
        "transformers==4.57.1",
        "numpy",
    ])
    .add_local_file(local_path="ragen_entry.py", remote_path="/root/ragen_entry.py")
)

app = modal.App(name="ragen-apo")


@app.function(gpu="A100-80GB", image=image, timeout=24 * 60 * 60)
def run():
    """Run RAGEN training on cloud GPU."""
    from ragen_entry import main
    main()


if __name__ == "__main__":
    # Run: modal run ragen_runner.py
    pass
