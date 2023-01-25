from typing import List

from pydantic import BaseModel


class DatasetConfig(BaseModel):
    dataset_name: str
    datasets_dir: str
    train_frac: float
    valid_frac: float
    neg_sampling_strat: str


class RegularizerConfig(BaseModel):
    type: str
    coeff: float


class ModelConfig(BaseModel):
    model_type: str = "complex"
    embedding_size: int = 256
    init_range: float = 0.1
    init_size: float = 1e-3


class TrainingConfig(BaseModel):
    epochs: int = 20
    batch_size: int = 256
    regularizers: List[RegularizerConfig] = []
    optimizer: str = "adam"
    learning_rate: float = 0.1
    loss_fn: str = "crossentropy"
    grad_accumulation_step: int = 1
    neg_sampling_rate: int = 10
    score_rhs: bool = True
    score_lhs: bool = True
    score_rel: bool = False
    eval_steps: int = 1
    device: int = 0
    random_seed: int = 1234
    outputs_dir: str = "outputs"


class Configs(BaseModel):
    dataset_config: DatasetConfig
    model_config: ModelConfig
    training_config: TrainingConfig
