import json

with open("pipeline.ipynb", "r") as f:
    nb = json.load(f)

if "metadata" not in nb:
    nb["metadata"] = {}

nb["metadata"]["kaggle"] = {
    "accelerator": "nvidiaTeslaT4",
    "isInternetEnabled": True,
    "language": "python",
    "sourceType": "notebook",
    "isGpuEnabled": True
}

with open("pipeline.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
