from RoADDataset import Dataset

road = Dataset(normalize=True)

print("Sous-ensembles :", list(road.sets.keys()))
for name, recordings in road.sets.items():
    print(name, "=", len(recordings), "séquence(s)")

print("Forme training[0] :", road.sets["training"][0].shape)