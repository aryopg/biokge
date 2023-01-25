from typing import List

from pydantic import BaseModel


class DatasetConfig(BaseModel):
    name: str


class ModelConfig(BaseModel):
    name: str


class TrainingConfig(BaseModel):
    random_seed: int
    training_loop: str
    epochs: int


class Configs(BaseModel):
    dataset_config: DatasetConfig
    model_config: ModelConfig
    training_config: TrainingConfig
