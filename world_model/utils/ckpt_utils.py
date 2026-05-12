"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from typing import Literal, Optional, Tuple
import string
import random
from pathlib import Path
from omegaconf import DictConfig
import wandb
from utils.print_utils import cyan
# from .huggingface_utils import download_from_hf

def is_run_id(run_id: str) -> bool:
    """Check if a string is a run ID."""
    return len(run_id) == 8 and run_id.isalnum()


def version_to_int(artifact) -> int:
    """Convert versions of the form vX to X. For example, v12 to 12."""
    return int(artifact.version[1:])


def download_latest_checkpoint(run_path: str, download_dir: Path) -> Path:
    api = wandb.Api()
    run = api.run(run_path)

    # Find the latest saved model checkpoint.
    latest = None
    for artifact in run.logged_artifacts():
        if artifact.type != "model" or artifact.state != "COMMITTED":
            continue

        if latest is None or version_to_int(artifact) > version_to_int(latest):
            latest = artifact

    # Download the checkpoint.
    download_dir.mkdir(exist_ok=True, parents=True)
    root = download_dir / run_path
    latest.download(root=root)
    return root / "model.ckpt"



def is_run_id(run_id: str) -> bool:
    """Check if a string is a run ID."""
    return len(run_id) == 8 and run_id.isalnum()


def generate_run_id() -> str:
    """Generate a random 8-character alphanumeric string."""
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(8))


def generate_unexisting_run_id(entity: str, project: str) -> str:
    """Generate a random 8-character alphanumeric string that does not exist in the project."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    existing_ids = {run.id for run in runs}
    while True:
        run_id = generate_run_id()
        if run_id not in existing_ids:
            return run_id


def parse_load(load: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse load into run_id and download option.
    (for load=xxxxxxxx in configurations)
    - If load_id is a run_id or path, return the run_id and None.
    - If load_id is of the form run_id:option, return run_id and option.
    - Otherwise, return None, None.
    """
    split = load.split(":")
    if 1 <= len(split) <= 2 and (is_run_id(split[0]) or Path(split[0]).exists()):
        return split[0], (split[1] if len(split) == 2 else None)
    return None, None


def version_to_int(artifact) -> int:
    """Convert versions of the form vX to X. For example, v12 to 12."""
    return int(artifact.version[1:])


def is_existing_run(run_path: str) -> bool:
    """Check if a run exists."""
    api = wandb.Api()
    try:
        _ = api.run(run_path)
        return True
    except wandb.errors.CommError:
        return False
    return False


def has_linked_checkpoint(run_path: str) -> bool:
    for file_name in Path(run_path).glob("*.ckpt"):
        if file_name.resolve().exists():
            return True
    return False




def retrive_checkpoint(
    run_path: str, checkpoint_dir: str, option: Literal["latest", "best"] = "latest"
):  
    run_path = run_path.replace("_eval", "") # if it's an eval run, we want to load from the corresponding training run.
    file_name = Path(checkpoint_dir) / Path(run_path) / f"{option}.ckpt"
    if file_name.resolve().exists():
        return file_name.resolve()
    else:
        print(f"No {option} model checkpoint found in {run_path}.")


def wandb_to_local_path(run_path: str) -> Path:
    return Path("outputs/downloaded") / run_path / "model.ckpt"
