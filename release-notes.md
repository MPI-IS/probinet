Release Notes:

# **ProbINet** - Release Notes for Version 1.0.2 🛠️✨

We’re excited to announce the first release of **probinet**, a Python package designed for **probabilistic network analysis**. This initial version introduces powerful tools and features to help users perform probabilistic network inference, generate synthetic networks, and evaluate models effectively.

---

## 🚀 Features

### 1. **Probabilistic Network Inference**
The package includes **five probabilistic models** for network inference, empowering users with flexible and robust tools for analyzing their network data. These models are:
1. MTCOV: https://www.nature.com/articles/s41598-020-72626-y
2. CRep: https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.023209
3. JointCRep: https://academic.oup.com/comnet/article/10/4/cnac034/6658441?login=true
4. DynCRep: https://iopscience.iop.org/article/10.1088/2632-072X/ac52e6
5. ACD: https://journalofbigdata.springeropen.com/articles/10.1186/s40537-022-00669-1

### 2. **Synthetic Network Generation**
Using the package's generative models (`synthetic`), users can:
- Fit parameters to real network data.
- Generate synthetic networks that resemble real-world networks, capturing their essential properties.

### 3. **Model Selection Module**
The built-in `model_selection` module enables:
- Automated selection of the best model for your data.
- Easy configuration and tuning of model parameters to improve performance.

### 4. **Performance Evaluation Metrics**
Evaluate and compare models effectively with the included performance metrics. These tools provide insight into the strengths and weaknesses of the applied probabilistic methods.

---

## 🔍 Downstream Tasks

After fitting a model to your data, ProbINet enables users to perform a variety of downstream tasks, including:

- **Community Detection**: Identifying tightly-knit groups within networks.
- **Reciprocity Estimation**: Measuring mutual connections within the network.
- **Anomaly Detection**: Detecting unusual or unexpected patterns in network data.
- **Link Prediction**: Predicting missing or future connections between nodes.

These powerful capabilities make ProbINet a versatile tool for analyzing and working with network data.

---

## 📖 Tutorials and Documentation 📚

- **Step-by-step tutorials** for every major feature of the package.
- **Real-world examples** on how to:
  - Fit models to data.
  - Generate synthetic networks.
  - Select the best parameters and models.
- Guides to performing downstream tasks like **community detection**, **reciprocity estimation**, **anomaly detection**, and **link prediction**.

---

## 🔧 Use Cases

**ProbINet** is ideal for researchers, data scientists, and engineers working with network data, such as:

- **Social networks**, e.g., interactions between users or groups.
- **Biological interaction networks**, e.g., gene or protein interaction networks.
- **Infrastructure/communication networks**, e.g., transportation, energy grids, or telecommunications.
- Synthetic data generation for **benchmarking** or **testing algorithms**.

---

## 🛠️ Getting Started

Install the package with:

```bash
pip install probinet (This wont't work until the package is up in PyPi, but soon)
```

Learn more by exploring the documentation and tutorials at:  
➡️ https://mpi-is.github.io/probinet/

---

## 🙌 Community and Feedback

This version is just the beginning, and we are eager to hear your feedback! 🚀  
If you encounter any issues or have feature requests, don’t hesitate to reach out or file an issue on our GitHub repository.

---

_Thank you for using ProbINet!_