from typing import List, Optional

from pydantic import BaseModel


class DatasetConfigs(BaseModel):
    datasets_dir: str
    tasks: List[str]
    train_frac: float
    valid_frac: float


class RegularizerConfigs(BaseModel):
    type: str
    coeff: float


class ModelConfigs(BaseModel):
    model_type: str = "complex"
    pretrained_path: Optional[str] = None
    pretrained_entity_ids_path: Optional[str] = None
    adjacency_matrix: Optional[str] = None
    hidden_size: Optional[int] = 256
    regularizers: Optional[List[RegularizerConfigs]] = []
    init_range: Optional[float] = 0.1
    init_size: Optional[float] = 1e-3
    dropout: Optional[float] = 0.0
    batch_size: Optional[int] = 256
    optimizer: Optional[str] = "adam"
    learning_rate: Optional[float] = 0.1
    loss_fn: Optional[str] = "crossentropy"
    grad_accumulation_step: Optional[int] = 1
    score_rhs: Optional[bool] = True
    score_lhs: Optional[bool] = True
    score_rel: Optional[bool] = False
    neg_sampling: Optional[str] = "none"
    neg_sampling_rate: Optional[int] = 10
    grad_clip: Optional[float] = 1


class EvaluationConfigs(BaseModel):
    eval_steps: int = 1
    K: List[int] = [1, 10, 100]


class TrainingConfigs(BaseModel):
    epochs: int = 20
    evaluation: EvaluationConfigs
    device: Optional[int] = 0
    random_seed: int = 1234
    outputs_dir: str = "outputs"


class TestingConfigs(BaseModel):
    epochs: int = 20
    freeze_embedding: bool = True
    device: Optional[int] = 0
    random_seed: int = 1234
    outputs_dir: str = "outputs"


class Configs(BaseModel):
    dataset_configs: DatasetConfigs
    model_configs: ModelConfigs
    training_configs: Optional[TrainingConfigs]
    testing_configs: Optional[TestingConfigs]
