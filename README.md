# Probabilistic Generative Model for Nets (PGM)

## Description

Welcome to the _Probabilistic Generative Models for Network Analysis_! Here, we host a Python
package that integrates
code from scientific publications, emphasizing probabilistic network analysis. Our package
streamlines tasks such as
community detection, anomaly detection, and synthetic data generation, making network analysis more
accessible and
cohesive.

## Installation

First, create a virtual environment and install the dependencies with pip:

```bash
python3 -m venv --copies venv
. venv/bin/activate
(venv) pip install -u pip # optional but always advised!
```

Then, you can install the package by running:

```bash
(venv) pip install .
```

## Project Structure

The package is structured as follows:

- **pgm/:** Main source code of the PGM package.
- **pgm/data/:** Contains input and model data files.
- **pgm/input/:** Scripts for generating and loading network data.
- **pgm/model/:** Implementation of different models like CRep.
- **pgm/output/:** Scripts for evaluating and processing output data.

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
