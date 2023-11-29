Welcome to the documentation for the Probabilistic Generative Models (``pgm``) Python package. This project is a
collaborative effort to consolidate state-of-the-art probabilistic generative modeling implementations from various
scientific publications. Our focus lies in advancing network analysis techniques with an emphasis on recent modeling
approaches that relax restrictive assumptions, enabling the modeling of joint distributions of data and latent
variables.

The PGM package is designed to be a comprehensive and user-friendly toolset for researchers and practitioners engaging
in probabilistic generative modeling for networks. Our goal is to provide a unified resource that brings together
diverse advances scattered across different code repositories. By doing so, we aim to simplify the use of machine
learning tasks on networks, including node clustering (community detection), anomaly detection, and the generation of
synthetic data from latent variables.

## Installation

To get started, follow these steps:

1. Create a virtual environment. For example, using ``venv``::

```bash
   python3 -m venv --copies venv
   . venv/bin/activate
   (venv) pip install -U pip # optional but always advised!
```

2. Install the ``pgm`` package by running::

```bash
   (venv) pip install .
```

## Usage

Run the PGM package as a whole with the following command::

```bash
    python main.py
```

## Tests

To run the tests::

```bash
    python -m unittest
```
