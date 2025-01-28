Contributing
============

Thank you for your interest in contributing to **ProbINet**! Your contributions help improve the
package and make it more useful for everyone. Please follow the steps below to get started.

Getting Started
---------------

- Fork this repository to your GitHub account: `https://github.com/MPI-IS/probinet <https://github.com/MPI-IS/probinet>`_. You can fork it by clicking the "Fork" button in the top-right corner of the page.
- Clone the forked repository to your local machine:

  .. code-block:: bash

     git clone https://github.com/<your-username>/probinet.git

- Navigate to the project directory:

  .. code-block:: bash

     cd probinet

- Add the original repository as a remote named ``upstream`` to keep your fork updated:

  .. code-block:: bash

     git remote add upstream https://github.com/MPI-IS/probinet.git

Setting Up the Development Environment
--------------------------------------

This project uses ``pyproject.toml`` for dependency management. To set up a development environment:

- Install the package with the ``.dev`` flag:

  .. code-block:: bash

     pip install ".[dev]"

- Verify the installation by running the following command:

  .. code-block:: bash

     pip show probinet

If the package is installed, this will display details about it, such as the version and installation location.

Syncing Your Fork
-----------------

Before starting work on a new feature or bug fix, ensure your fork is up to date with the original repository:

- Fetch the latest changes from the ``upstream`` repository:

  .. code-block:: bash

     git fetch upstream

- Update your local ``main`` branch:

  .. code-block:: bash

     git checkout main
     git merge upstream/main

Making Changes
--------------

- Create a new branch for your contribution:

  .. code-block:: bash

     git checkout -b feature/your-feature-name

- Make your changes in this branch. Ensure the code is:

  - Well-documented.
  - Aligned with the existing code style.

- Add or update unit tests for your changes. You can see the existing tests in the ``tests`` directory.

Running Tests
-------------

Tests are written using Python's built-in ``unittest`` framework.

- Run all tests to verify your changes:

  .. code-block:: bash

     python -W ignore -m unittest discover

Submitting Your Contribution
----------------------------

- Commit your changes with a clear and concise message:

  .. code-block:: bash

     git commit -m "Add description of your changes"

- Push your changes to your fork:

  .. code-block:: bash

     git push origin feature/your-feature-name

- Open a Pull Request (PR) to the **original repository**. Include:

  - A detailed explanation of your changes.
  - The issue number your PR addresses (if applicable).
  - Any additional context or screenshots.

  You can view all open and merged Pull Requests `here <https://github.com/MPI-IS/probinet/pulls>`_.


Code of Conduct
---------------

By contributing to this repository, you agree to follow our `Code of Conduct
<https://policies.python.org/python.org/code-of-conduct/>`_.

We appreciate your contributions and will review your Pull Request promptly!
