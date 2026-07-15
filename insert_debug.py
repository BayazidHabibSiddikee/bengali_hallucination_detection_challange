import json

with open("pipeline.ipynb", "r") as f:
    nb = json.load(f)

debug_cell = {
    "cell_type": "code",
    "metadata": {},
    "source": [
        "import os\n",
        "print('--- DIAGNOSTICS ---')\n",
        "if os.path.exists('/kaggle/input'):\n",
        "    for root, dirs, files in os.walk('/kaggle/input'):\n",
        "        print(root, dirs, files)\n",
        "print('-------------------')\n"
    ],
    "outputs": [],
    "execution_count": None
}

nb["cells"].insert(1, debug_cell)

with open("pipeline.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
