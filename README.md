# Probabilistic Generative Model for Nets (PGM)

## Description

Welcome to the _Probabilistic Generative Models for Network Analysis_! Here, we host a Python package that integrates code from scientific publications, emphasizing probabilistic network analysis. Our package streamlines tasks such as community detection, anomaly detection, and synthetic data generation, making network analysis more accessible and cohesive.


## Installation

First, create a virtual environment and install the dependencies with pip:

```bash
python3 -m venv --copies venv
. venv/bin/activate
(venv) pip install -u pip # optional but always advised!
(venv) pip install .
```
Then, you can install the dependencies by running:

```bash
pip install -r requirements.txt
```

## Project Structure

The package is structured as follows:

```bash
prob-gen-model-for-nets/
├── src/
│   └── pgm/
│       ├── data/
│       │   ├── input/
│       │   │   ├── setting111.yaml
│       │   │   ├── syn111.dat
│       │   │   └── theta_gt111.npz
│       │   └── output/
│       │       ├── setting_CRep.yaml
│       │       ├── theta_CRep.npz
│       │       └── theta_test.npz
│       ├── input/
│       │   ├── generate_network.py
│       │   ├── loader.py
│       │   ├── preprocessing.py
│       │   ├── setting_CRep0.yaml
│       │   ├── setting_CRepnc.yaml
│       │   ├── setting_CRep.yaml
│       │   ├── setting_syn_data.yaml
│       │   ├── statistics.py
│       │   └── tools.py
│       ├── model/
│       │   ├── CRep.py
│       │   ├── cv_CRep.py
│       │   ├── cv.py
│       ├── output/
│       │   ├── evaluate.py
├── tests/
│   └── test.py
```

### Directories

- **src/pgm/:** Main source code of the PGM package.
- **src/pgm/data/:** Contains input and output data files.
- **src/pgm/input/:** Scripts for generating and loading network data.
- **src/pgm/model/:** Implementation of different models like CRep.
- **src/pgm/output/:** Scripts for evaluating and processing output data.

## Usage

To run the package as a whole, you can use the following command:
```bash
python main.py
```

## Tests

To run the tests:

```bash
python -m unittest discover tests
```
