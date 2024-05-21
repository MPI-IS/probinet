# Probabilistic Generative Models

Welcome to the documentation for the Probabilistic Generative Models (``pgm``) Python package. This project is a
collaborative effort to consolidate state-of-the-art probabilistic generative modeling implementations from various
scientific publications. Our focus lies in advancing network analysis techniques with an emphasis on recent modeling
approaches that relax the restrictive conditional independence assumption, enabling the modeling of joint 
distributions of network data.  

The ``pgm`` package is designed to be a comprehensive and user-friendly toolset for researchers and practitioners 
interested in modeling network data through probabilistic generative approaches. Our goal is to provide a 
unified resource that brings together different advances scattered across many code repositories. 
By doing so, we aim not only to enhance the usability of existing models, but also to facilitate the comparison 
of different approaches. Moreover, through a range of tutorials, we aim at simplifying the use of these methods 
to perform inferential tasks, including the prediction of missing network edges, node clustering (community detection), 
anomaly detection, and the generation of synthetic data from latent variables.

## Installation

This package requires Python 3.10. Please ensure you have this version before proceeding with the installation.

To get started, follow these steps:

1. Create a virtual environment. For example, using ``venv``:

```bash
   python3.10 -m venv --copies venv
   . venv/bin/activate
   (venv) pip install -U pip # optional but always advised!
```

2. Install the ``pgm`` package by running:

```bash
   (venv) pip install .
```

## Usage

Run the `pgm` package as a whole with the `run_model` command. A list of the parameters that can be passed as arguments is available by running:

```bash
    run_model --help
```

To run a specific model, pass the model name as an argument with the `-a` flag. For example, to run the `CRep` model, use:

```bash
    run_model -a CRep
```
The `run_model` command can be run at different logging levels. To run the command with the `DEBUG` level, use:

```bash
    run_model -a CRep -d
```

## Tests

To run the tests:

```bash
    python -m unittest
```
