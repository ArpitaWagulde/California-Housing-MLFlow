## Getting Started

### 1. Environment Setup
You can set up a virtual environment and install the necessary dependencies using the following commands:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run MLFlow

Run following command in a new terminal, in virtual environment
```bash
mlflow ui
```

### 3. Run Training script

```bash
python src/train.py
```

### 4. Run validate and promote script

```bash
python src/promote.py
```

### 4. Observe metrics

Open http://127.0.0.1:5000 to observe the model runs, model versions and compare metrics