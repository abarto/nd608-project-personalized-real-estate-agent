## nd608 - Project: Personalized Real Estate Agent

### Introduction

This is repo contains my implementation for Udacity's Generative AI Nanodregree "Project: Personalized Real Estate Agent.

The project uses [`poetry`](https://python-poetry.org/) for dependency management. Use `poetry install --no-root` on your environment to install the required python packages.

The code uses [`python-dotenv`](https://pypi.org/project/python-dotenv/) to load an `.env` file (not provided) from the same directory to set the `OPENAI_API_KEY` value environment variable. If this library is not available within Udacity's workspace, the client's `api_key` parameter will have to be set manually.

### Contents

| File/Folder | Description |
|-------------|-------------|
| [`Udacity-Project-Environment`](Udacity-Project-Environment) | The empty project environment provided by Udacity. |
| [`Generate-Real-Estate-Listings.ipynb`](Generate-Real-Estate-Listings.ipynb) | Jupyter Notebook where we generate the synthetic real estate listings as well as setting up the [LanceDb](https://lancedb.com/) vector database with the listings' embeddings. |
| [`Generate-Real-Estate-Listings.html`](Generate-Real-Estate-Listings.html) | Static HTML version of [`Generate-Real-Estate-Listings.ipynb`](Generate-Real-Estate-Listings.ipynb). |
| [`models.py`](models.py) | Python module with models shared between [`Generate-Real-Estate-Listings.ipynb`](Generate-Real-Estate-Listings.ipynb) and the `app.py` module. |
| [`data`](data) | Folder containing the [LanceDb](https://lancedb.com/) data files as well as `pickle` files used as intermediate storage during the real estate listings generation. |
| [`pyproject.toml`](pyproject.toml) | [`poetry`](https://python-poetry.org/) project specification. |
| [`poetry.lock`](poetry.lock) | [`poetry`](https://python-poetry.org/) dependency locks. |
