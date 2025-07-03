# 2025-07-mle-workshop-training

## Day 1: Training
### How to install UV
- just run 'curl -LsSf https://astral.sh/uv/0.7.19/install.sh | sh'
- change directory into it vi 'cd day_1'
- run 'uv init -- python 3.10'
- run 'uv sync'
### Dowload the notebook:

- 'cd notebooks'
-'wget "https://raw.githubusercontent.com/alexeygrigorev/ml-engineering-contsructor-workshop/refs/heads/main/01-train/duration-prediction-starter.ipynb"'

### install dependencies
- 'uv add scikit-learn==1.2.2 pandas pyarrow'
- -install jupyter 'uv add --dev jupyter seaborn'

### run notebook
- uv run  jupyter notebook

### Convert notebook into script:
- uv run jupyter nbconvert --to=script notebooks/duration-prediction-starter.ipynb
- Create a new folder 'duration_prediction' and move the file into that folder and renaime it into 'train.py'