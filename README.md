# ProbINet

*This is a beta release currently being tested by users. Your feedback is valuable as we work 
towards finalizing the package!*

Welcome to the documentation for the **Prob**abilistic **I**nference on **Net**works 
(``ProbINet``) Python 
package. This project is a
collaborative effort to consolidate state-of-the-art probabilistic generative modeling implementations from various
scientific publications. Our focus lies in advancing network analysis techniques with an emphasis on recent modeling
approaches that relax the restrictive conditional independence assumption, enabling the modeling of joint 
distributions of network data.  

The ``ProbINet`` package is designed to be a comprehensive and user-friendly toolset for 
researchers and practitioners 
interested in modeling network data through probabilistic generative approaches. Our goal is to provide a 
unified resource that brings together different advances scattered across many code repositories. 
By doing so, we aim not only to enhance the usability of existing models, but also to facilitate the comparison 
of different approaches. Moreover, through a range of tutorials, we aim at simplifying the use of these methods 
to perform inferential tasks, including the prediction of missing network edges, node clustering (community detection), 
anomaly identification, and the generation of synthetic data from latent variables.

## Installation

This package requires Python 3.10. Please ensure you have this version before proceeding with the installation.

To get started, follow these steps:

1. Create a virtual environment. For example, using ``venv``:

```bash
   python3.10 -m venv --copies venv
   . venv/bin/activate
   (venv) pip install -U pip # optional but always advised!
```

2. Install the ``ProbINet`` package by running:

```bash
   (venv) pip install .
```

## Usage

Run the ``ProbINet`` package as a whole with the `run_model` command. A list of the parameters that can be passed as arguments is available by running:

```bash
    run_model --help
```

To run a specific model, pass the model name as an argument. The available models are: `CRep`, `JointCRep`, `MTCOV`, `DynCRep`, and `ACD`. For example, to run the `CRep` model, use: 

```bash
    run_model -a CRep
```

To see the specific options for a model, use the `-h` flag. For example, to see the options for the `CRep` model, use:  

```bash 
    run_model CRep -h 
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

## Documentation

The documentation can be built with *Sphinx* by running:

```bash
    cd doc
    make html
```

The tutorials are then displayed in the left sidebar of the generated HTML documentation. They 
can also be accessed directly from the [tutorials](doc/source/tutorials) folder.

## Authors

- [Diego Baptista Theuerkauf](https://github.com/diegoabt)
- [Jean-Claude Passy](jean-claude.passy@tuebignen.mpg.de)


## License

This project is licensed under the GNU GPL version 3 - see the [LICENSE](LICENSE.md) file for 
details.


## Copyright

© 2019, Max Planck Society / Software Workshop - Max Planck Institute for Intelligent Systems
