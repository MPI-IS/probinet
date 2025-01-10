# ProbINet

> **BETA RELEASE**:
This is a *beta release* currently being tested by users.
Your feedback is valuable as we work towards finalizing the package!

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


This package requires Python 3.10 or higher. Please ensure you have one of these versions before proceeding with the installation.
To get started, follow these steps:

1. Clone the repository and navigate to the `probinet` directory:

```bash
   git clone https://github.com/MPI-IS/probinet.git
   cd probinet
````

2. Create a virtual environment. For example, using ``venv``:

```bash
   python3 -m venv --copies venv
   . venv/bin/activate
   (venv) pip install -U pip # optional but always advised!
```

3. Install the ``ProbINet`` package by running:

```bash
   (venv) pip install .
```

## Usage

Run the ``ProbINet`` package as a whole with the `run_probinet` command. This command can be run 
from any directory after the package is installed.

A list of the parameters that can be passed as arguments is available by running:

```bash
    run_probinet --help
```

To run a specific model, pass the model name as an argument. The available models are: `CRep`, `JointCRep`, `MTCOV`, `DynCRep`, and `ACD`. For example, to run the `CRep` model, use:

```bash
    run_probinet CRep
```

To see the specific options for a model, use the `-h` flag. For example, to see the options for the `CRep` model, use:

```bash
    run_probinet CRep -h
```

The `run_probinet` command can be run at different logging levels. To run the command with the `DEBUG` level, use:

```bash
    run_probinet CRep -d
```

To set arguments with double dashes (e.g., `--convergence_tol`), include them in the command line 
as follows:

```bash
    run_probinet CRep --convergence_tol 0.1
```

Some commands can also be executed using shorter versions of the arguments. For example, the 
`--convergence_tol` argument can be shortened to `-tol`. For example:

```bash
    run_probinet CRep -tol 0.1
```
These shorter versions can be found in the help message of each model.

## Tests

To run the tests:

```bash
    python -m unittest
```

## Documentation

The documentation can be built with *Sphinx*. To install it, run:

```bash
    pip install sphinx
```

To build the documentation, run:

```bash
    cd docs
    make html
```

The documentation will be available in the [docs/build/html](docs/build/html) directory.

## Tutorials

The tutorials are available in the [docs/source/tutorials](docs/source/tutorials) directory. Each tutorial is a Jupyter 
notebook that can be run in a Jupyter environment. 

## Authors

- [Diego Baptista Theuerkauf](https://github.com/diegoabt)
- [Jean-Claude Passy](jean-claude.passy@tuebignen.mpg.de)

The authors of the original implementations integrated to this packages are:

- [Martina Contisciani](https://github.com/mcontisc) 
- [Hadiseh Safdari](https://github.com/hds-safdari) 
- [Caterina De Bacco](https://cdebacco.com/) 

See the references in the documentation for more details.

## Contributing

Would you like to contribute to the development of **ProbINet**? Contributions are welcome and 
appreciated! You can find detailed information on how to get started here: [Contributing Guide](https://mpi-is.github.io/probinet/contributing.html).


## License

This project is licensed under the GNU GPL version 3 - see the [LICENSE](LICENSE.md) file for
details.


## Copyright

© 2024, Max Planck Society / Software Workshop - Max Planck Institute for Intelligent Systems
