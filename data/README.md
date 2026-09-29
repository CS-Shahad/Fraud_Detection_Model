# Data

This project uses **PaySim**, "Synthetic Financial Datasets For Fraud Detection":
<https://www.kaggle.com/datasets/ealtaf/paysim1>. It is about 470 MB, so it is not stored in the repository.

You don't need to download it by hand: the code fetches it from Kaggle with `kagglehub` the first time you
run it. To use a copy you already have, either:

- put `PS_20174392719_1491204439457_log.csv` in this folder, or
- pass the path: `python -m fraud_detection.train --data path/to/file.csv`

**Source:** E. A. Lopez-Rojas, A. Elmir and S. Axelsson, "PaySim: A financial mobile money simulator for
fraud detection", *28th European Modeling and Simulation Symposium (EMSS)*, 2016.
