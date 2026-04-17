import numpy as np
import os

print("--- LOADER SCRIPT ---")

# load the features we saved
f = np.load("data/outputs/features/features.npy")
l = np.load("data/outputs/features/labels.npy")
idx = np.load("data/outputs/features/indices.npy")

print("Features shape: ", f.shape)
print("Labels shape  : ", l.shape)
print("Unique labels : ", np.unique(l, return_counts=True))

print("Ready for training.")