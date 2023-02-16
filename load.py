import torch

model = torch.load("checkpoint_best.pt", map_location="cpu")["model"][0]
print(model)

print(model["_entity_embedder._embeddings.weight"].shape)
print(model["_relation_embedder._embeddings.weight"].shape)
