import yaml
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, FilePath, DirectoryPath


# --- Nested Models ---
class WeightsPaths(BaseModel):
    pvt_v2_b2: FilePath
    pvt_v2_b4: FilePath
    pvt_v2_b5: FilePath


# --- Main Config ---
class Config(BaseModel):

    # Training
    batch_size: int = Field(..., gt=0)
    batch_size_valid: int = Field(..., gt=0)
    epochs: int = Field(..., gt=0)
    lr: float = Field(..., gt=0)
    weight_decay: float = Field(..., ge=0)
    rand_seed: int = 42
    precisionHigh: bool = False   # safer default

    # Dataset
    train_dir: DirectoryPath
    test_dir: DirectoryPath
    load_all: bool = False

    # Model
    backbone: str
    img_size: int = Field(..., gt=0)
    lateral_channels: List[int]
    resume: Optional[str] = None
    compile: bool = False
    use_frequency_branch: bool = False
    frequency_mode: str = "fft"
    use_amp: bool = True

    # GPU
    device_ids: List[int]
    multi_GPU: bool = False

    # Saving
    save_model_dir: str
    name: str
    save_last: int = Field(..., ge=0)
    save_step: int = Field(..., gt=0)

    # Augmentation
    preproc_methods: List[str]

    # Paths
    weights: WeightsPaths

    # System
    num_workers: int = Field(..., ge=0)

    # --- Helpers ---
    @property
    def is_ddp(self) -> bool:
        return self.multi_GPU and len(self.device_ids) > 1

    @property
    def save_path(self) -> Path:
        path = Path(self.save_model_dir) / self.name
        path.mkdir(parents=True, exist_ok=True)
        return path


# --- Loader ---
def load_config(config_path: str = "config.yaml") -> Config:
    """Load YAML safely with UTF-8 and normalize values."""

    # ✅ FIX 1: Encoding issue
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # ✅ FIX 2: Normalize resume
    if config_dict.get("resume") == "":
        config_dict["resume"] = None

    return Config(**config_dict)
