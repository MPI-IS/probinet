---
title: 'ProbINet: Bridging Usability Gaps in Probabilistic Network Analysis'

tags:
  - Python
  - network science
  - probabilistic modeling
  - community detection
  - anomaly detection
  - synthetic data generation
authors:
  - name: Diego Baptista
    orcid: 0000-0003-2994-0138
    #equal-contrib: true
    affiliation: "1, 2"
  - name: Martina Contisciani
    #corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - name: Caterina De Bacco
    #equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 4
  - name: Jean-Claude Passy
    affiliation: 1
affiliations:
  - name: Max Planck Institute for Intelligent Systems, Tübingen, Germany.
    index: 1
  - name: Graz University of Technology, Graz, Austria.
    index: 2
  - name: Central European University, Vienna, Austria.
    index: 3
  - name: Delft University of Technology, Delft, Netherlands.
    index: 4
date: 22 January 2025
bibliography: paper.bib

---

# Summary

**Prob**abilistic **I**nference on **Net**works (ProbINet) is a Python package that provides a 
unified framework to perform probabilistic inference on networks, enabling researchers and practitioners 
to analyze and model complex network data. The package integrates code implementations from several scientific publications, supporting tasks such as community detection, anomaly detection, and synthetic data generation using latent variable models. It is designed to simplify the use of cutting-edge techniques in network analysis by providing a cohesive and user-friendly interface. 

# Statement of need

Network analysis is central to disciplines such as social sciences, biology, and fraud detection, where understanding relationships is essential. Probabilistic generative models 
[@safdari2021generative; @contisciani2022community; @safdari2022anomaly; @safdari2022reciprocity; @contisciani2020community
] reveal hidden patterns, detect communities, identify anomalies, and generate synthetic data. Their broader use is limited by fragmented implementations that hinder comparisons and reproducibility. 
ProbINet addresses this gap by unifying recent approaches in a single framework, improving accessibility and usability across disciplines. 

ProbINet stands out among network analysis tools. Graph-tool [@peixoto_graph-tool_2014] provides community detection and general graph analysis tools, but it uses a different model family than our mixed-membership framework and does not account for reciprocity.  CDlib [@rossetti_cdlib_2019] offers detection algorithms and evaluation routines, but ProbINet extends this with probabilistic MLE models, optional node attributes, and anomaly detection. pgmpy [@ankan_pgmpy_2024] focuses on Bayesian network structure learning, while ProbINet uncovers latent patterns like communities and reciprocity.

# Main features

ProbINet offers a feature-rich framework to perform inference on networks using probabilistic 
generative models.  Key features include:

- **Diverse Network Models**: Integration of generative models for various network types
  and goals (see table below).

- **Synthetic Network Generation**: Ability to generate synthetic networks that closely resemble real ones for further analyses (e.g., testing hypotheses).

- **Simplified Parameter Selection**: A cross-validation module to optimize key parameters, providing performance results in a clear dataframe.

- **Rich Set of Metrics for Analysis**:  Advanced metrics (e.g., F1 scores, Jaccard index) for link and covariate prediction performance.

- **Powerful Visualization Tools**: Functions for plotting community memberships and performance metrics.

- **User-Friendly Command-Line Interface**: An intuitive interface for easy access.

- **Extensible and Modular Codebase**: Future integration of additional models possible.


| **Algorithm's Name**&nbsp; | **Description**                                                                                                         | **Network Properties**                                |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| **CRep**                   | Models directed networks with communities and reciprocity [@safdari2021generative].                                     | Directed, Weighted, Communities, Reciprocity          |
|                            |                                                                                                                         |                                                       |
| **JointCRep**              | Captures community structure and reciprocity with a joint edge distribution [@contisciani2022community].                | Directed, Communities, Reciprocity                    |
|                            |                                                                                                                         |                                                       |
| **DynCRep**                | Extends CRep for dynamic networks [@safdari2022reciprocity].                                                            | Directed, Weighted, Dynamic, Communities, Reciprocity |
|                            |                                                                                                                         |                                                       |
| **ACD**                    | Identifies anomalous edges and node community memberships in weighted networks [@safdari2022anomaly].                                     | Directed, Weighted, Communities, Anomalies            |
|                            |                                                                                                                         |                                                       |
| **MTCOV**                  | Extracts overlapping communities in multilayer networks using topology and node attributes [@contisciani2020community]. | Weighted, Multilayer, Attributes, Communities         |

The **Usage** section below illustrates these features with a real-world example.

# Usage

## Example: Analyzing a Social Network with ProbINet

This section shows how to use ProbINet to analyze a social network of 31 students and 100 
directed edges representing friendships in a small Illinois high school [@konect:coleman]. We analyze the network using JointCRep in ProbINet to infer latent variables, assuming communities and reciprocity drive tie formation, a reasonable assumption for friendship relationships.

### Steps to Analyze the Network with ProbINet

With ProbINet, you can load network data as an edge list and select an algorithm (e.g., JointCRep), 
fit the model to extract latent variables, and analyze results like soft community memberships, 
which show how nodes interact across communities.  This is exemplified in Figure 1. On the left, a 
network representation of the input data is displayed alongside the lines of code required for 
its analysis using ProbINet. The result is shown on the right, where nodes are colored according to their inferred soft community memberships, while edge thickness and color intensity represent the inferred probability of edge existence. 

![Usage of ProbINet on a social network. (Top-left) A network representation of the input data.  (Bottom-left) A snapshot of the code used. (Right) The resulting output.](figures/example.png)

For more tutorials and use cases, see the [package documentation](https://mpi-is.github.io/probinet/).

# Running Times of Algorithms

The table below summarizes algorithm runtimes on the tutorial data.
**N** and **E** represent the number of nodes and edges, respectively.
Edge ranges indicate variation across layers or time steps.
**L/T** indicates the number of layers or time steps,
and **K** represents the number of communities.

| **Algorithm** | **N** | **E**    | **L/T** | **K** | **Time (mean ± std, in seconds)** |
|---------------|-------|----------|---------|-------|-----------------------------------|
| **CRep**      | 600   | 5512     | 1       | 3     | 3.00 ± 0.35                       |
| **JointCRep** | 250   | 2512     | 1       | 2     | 3.81 ± 0.69                       |
| **DynCRep**   | 100   | 234-274  | 5       | 2     | 1.48 ± 0.06                       |
| **ACD**       | 500   | 5459     | 1       | 3     | 27.8 ± 3.2                        |
| **MTCOV**     | 300   | 724-1340 | 4       | 2     | 1.51 ± 0.14                       |


These benchmarks were performed on a 12th Gen Intel Core i9-12900 CPU, using `hyperfine` [@Peter_hyperfine_2023] and 10 runs.
Runs required small amounts of RAM (less than 1 GB).

# Acknowledgements

We thank the contributors of the integrated publications and Kibidi Neocosmos, Valkyrie Felso, and Kathy Su for their feedback.

# References
