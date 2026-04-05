import numpy as np

features = np.load("data/features/features.npy")
labels   = np.load("data/features/labels.npy")
indices  = np.load("data/features/indices.npy")

print("Features shape:", features.shape)
print("Labels shape  :", labels.shape)
print("Unique labels :", np.unique(labels, return_counts=True))