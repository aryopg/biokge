from typing import List

from pydantic import BaseModel


class DatasetConfig(BaseModel):
    name: str


class ModelConfig(BaseModel):
    name: str
    embedding_dim: int


class TrainingConfig(BaseModel):
    random_seed: int
    training_loop: str
    learning_rate: float
    num_negs_per_pos: int
    epochs: int


class Configs(BaseModel):
    dataset_config: DatasetConfig
    model_config: ModelConfig
    training_config: TrainingConfig
