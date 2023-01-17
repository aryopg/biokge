from typing import List, Optional

from pydantic import BaseModel


class DatasetConfigs(BaseModel):
    dataset_name: str
    datasets_dir: Optional[str]


class RegularizerConfigs(BaseModel):
    type: str
    coeff: float


class ModelConfigs(BaseModel):
    model_type: str = "complex"
    hidden_size: int = 256
    regularizers: List[RegularizerConfigs] = []
    init_range: float = 0.1
    init_size: float = 1e-3
    dropout: float = 0.0
    batch_size: int = 256
    optimizer: str = "adam"
    learning_rate: float = 0.1
    loss_fn: str = "crossentropy"
    grad_accumulation_step: int = 1


class TrainingConfigs(BaseModel):
    epochs: int = 20
    eval_steps: int = 1
    device: int = 0
    random_seed: int = 1234
    outputs_dir: str = "outputs"


class Configs(BaseModel):
    dataset_configs: DatasetConfigs
    model_configs: ModelConfigs
    training_configs: TrainingConfigs
