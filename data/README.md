# Data

This project uses **PaySim**, "Synthetic Financial Datasets For Fraud Detection":
<https://www.kaggle.com/datasets/ealtaf/paysim1>. It is about 470 MB, so it is not stored in the repository.

Kaggle requires signing in to download it. Either:

- set a Kaggle API token (`KAGGLE_API_TOKEN`, or `KAGGLE_USERNAME` and `KAGGLE_KEY`) and the code downloads
  it with `kagglehub` the first time you run it,
- put the CSV, or the `archive.zip` Kaggle gives you, in this folder, or
- pass the path: `python -m fraud_detection.train --data path/to/file.csv`

**Source:** E. A. Lopez-Rojas, A. Elmir and S. Axelsson, "PaySim: A financial mobile money simulator for
fraud detection", *28th European Modeling and Simulation Symposium (EMSS)*, 2016.
