# Fastbook for Apple Silicon - 2026

- Install Miniconda
- conda create -n fastai python=3.12
- cd fastbook/
- conda activate fastai
- pip install fastbook torch torchvision torchaudio fastai fastprogress ipywidgets jupyter ipykernel
- pip freeze > requirements-apple.txt
- python -m ipykernel install --user --name fastai_env --display-name="fastai env"
