---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: pgm
    language: python
    name: pgm
---

# Tutorial: Generation of synthetic networks using the `CRep` algorithm

In this tutorial, we show how to use the _Probabilistic Generative Models_ (`pgm`) package for generating synthetic network data. 

We use the `CRep` (**C**ommunity and **Re**ci**p**rocity) algorithm, which is a probabilistic generative method designed to model directed networks. The main assumption of this approach is that communities and reciprocity are the main mechanisms for tie formation. 


As a first step, let's configure the logger to show the information about the execution of the algorithms.

```python
# Import the logging module
import logging

# Get the root logger and set its level to INFO
logging.getLogger().setLevel(logging.INFO)
```

## Generating a synthetic network using the `CRep` algorithm


The first step in our network generation process consists of setting the configuration file. This file contains the model parameters needed to generate the data. As explained in reference [1], the `CRep` algorithm has several parameters that can be set, including the  number of nodes `N`, the number of communities `K`, the reciprocity coefficient `eta`, and the type of community `structure`. Instead of setting these parameters manually, we can use the configuration file to illustrate the model's basic needs.

```python
# Import the `open_binary` function from the `importlib.resources` module
# This function is used to open a binary file included in a package
from importlib.resources import open_binary

# Import the `yaml` module to convert the data from a YAML formatted string into a Python dictionary
import yaml

# Define the path to the configuration file for the `CRep` algorithm
config_path = 'setting_syn_data_CRep.yaml'

# Open the configuration file for the `CRep` algorithm
with open_binary('pgm.data.model', config_path) as fp:
    # Load the content of the configuration file into a dictionary
    synthetic_configuration = yaml.load(fp, Loader=yaml.Loader)
```

```python
synthetic_configuration
```

As we can see in the dictionary, the reciprocity coefficient `eta` is set to $0.5$, meaning that the network will have a moderate level of reciprocity. Nevertheless, we are interested in generating a network with a higher level of reciprocity, thus we increase this value to $0.8$. We also modify some details regarding the output of the algorithm.

```python
# Increase the reciprocity coefficient
synthetic_configuration['eta'] = 0.8
# The flag `output_parameters` determines whether the model parameters should be saved to a file
synthetic_configuration['output_parameters'] = False
# The flag `output_adj` determines whether the adjacency matrix should be saved to a file
synthetic_configuration['output_adj'] = True
# The argument `outfile_adj` determines the name of the file for the adjacency matrix
synthetic_configuration['outfile_adj'] = 'syn_dataframe.dat'
# The argument `out_folder` determines the output folder for the adjacency matrix
synthetic_configuration['out_folder'] = 'tutorial_outputs/CRep_synthetic/'
```

Once the parameters are set, we can generate a synthetic network using the `GM_reciprocity` class.

```python
# Load the `GM_reciprocity` class from the `pgm.input.generate_network` module
from pgm.input.generate_network import GM_reciprocity

# Define the class `gen` as an instance of the `GM_reciprocity` class using the configuration parameters
gen = GM_reciprocity(**synthetic_configuration)
```

We can check that the model parameters have been set correctly as attributes of the `gen` object.

```python
gen.__dict__
```

We can now generate a synthetic network using the `reciprocity_planted_network` method, which follows the assumptions of the `CRep` algorithm. Notice that a network generated with this method is directed and weighted. The function returns a MultiDiGraph NetworkX object `G` and its adjacency matrix `A`.

```python
# Generate the network using the `reciprocity_planted_network` method
G, A = gen.reciprocity_planted_network()
```

Notice that, although we set the reciprocity coefficient to $0.8$, the actual network reciprocity is $0.638$. This mismatch is because `CRep` generates weighted networks. By considering the edge weights, the actual network reciprocity becomes $0.827$, pretty close to the desired one. 


We can now inspect how the network looks like.

```python
import matplotlib.pyplot as plt

# Plot the adjacency matrix
plt.imshow(A.toarray(), cmap='Greys', interpolation='none')
plt.colorbar()
plt.show()
```

As we can see, the generated synthetic network reflects an assortative community structure as imposed with the parameter `structure`. In this setting, nodes tend to connect more within their own communities than with nodes from other communities, resulting in higher edge densities within the diagonal blocks of the adjacency matrix. 


When `output_adj=True`, the function saves the network edges as a dataframe into the `syn_dataframe.dat` file in the output folder. The first and the second columns of the dataframe describe the source and the target nodes, respectively, and the last column represents the weight of their interactions. 

```python
import pandas as pd

# Load the dataframe
df = pd.read_csv(synthetic_configuration['out_folder'] + synthetic_configuration['outfile_adj'], sep=' ')
# Print the first 5 rows of the dataframe 
df.head()
```

In the next section, we will use the `pgm` package to analyze the network and extract the community structure and reciprocity coefficient.


## Analyzing the network using the `pgm` package

First, we start by importing the data using the `pgm` package. This means, we will load the data 
from the `syn_dataframe.dat` file and generate the adjacency matrices needed to run the `CRep` algorithm.

```python
from pgm.input.loader import import_data
from pathlib import Path

# Define the names of the columns in the input file that 
# represent the source and target nodes of each edge.
ego = 'source'
alter = 'target'

# Set the `force_dense` flag to False
force_dense = False

# Set the `binary` flag to False to load the edge weights
binary = False

# Call the `import_data` function to load the data from the input file
A, B, B_T, data_T_vals = import_data(Path(synthetic_configuration['out_folder']) / synthetic_configuration['outfile_adj'],
                                     ego=ego,
                                     alter=alter,
                                     force_dense=force_dense,
                                     binary=binary,
                                     header=0)
```

The `import_data` function prints some information about the data, such as the number of nodes and edges, as well as the average degree and the reciprocity. These statistics confirm we uploaded the right dataset, that is the one we just generated synthetically. 


Once the network is loaded, we can give it as input to the `CRep` algorithm to obtain estimates of the latent
variables describing the communities and the reciprocity. To do so, we need to set the configuration file for the `CRep` algorithm.

```python
# Set the algorithm to 'CRep'
algorithm = 'CRep'

# Define the path to the configuration file for the `CRep` algorithm
config_path = 'setting_' + algorithm + '.yaml'
```

We load the configuration file using the data files in the `pgm` package instead of a relative path to it.

```python
# Open the configuration file for the `CRep` algorithm
with open_binary('pgm.data.model', config_path) as fp:
    conf = yaml.load(fp, Loader=yaml.Loader)
```

```python
# Print the configuration file
print(yaml.dump(conf))
```

The previous file shows the parameters actually needed to _run_ the model. These parameters set the algorithms basic needs to work.  


Now, let's change the path to the output folder, so we can save the results into the same folder 
where the input data is located. 

```python
# Set the output folder for the `CRep` algorithm
conf['out_folder'] = synthetic_configuration['out_folder']

# Set the end file for the `CRep` algorithm
conf['end_file'] = '_' + algorithm
```

## Running the Model
Finally, we are ready to run the `CRep` model! The way this works is in a two-step process: <br>
i) first, we call the `CRep` class, which initializes the model; <br>
ii) then, we call the `fit` method, which runs the algorithm. 


```python
# Import the `CRep` class from the `pgm.model.crep` module
from pgm.model.crep import CRep

# Get the list of nodes from the first graph in the list `A`
nodes = A[0].nodes()

# Create an instance of the `CRep` class
model = CRep()

# Print all the attributes of the `CRep` instance
# The `__dict__` attribute of an object is a dictionary containing 
# the object's attributes.
print(model.__dict__)
```

Model created! Now, we can run the model using the `fit` method. As mentioned before, this method takes as input the data, and the configuration parameters. 

```python
# Import the `time` module
import time

# Import the `numpy` module
import numpy as np

# Print a message indicating the start of the `CRep` algorithm
print(f'\n### Run {algorithm} ###')

# Get the current time
time_start = time.time()

# Run the `CRep` model
inferred_parameters = model.fit(data=B,
              data_T=B_T,
              data_T_vals=data_T_vals,
              nodes=nodes,
              **conf)

# Print the time elapsed since the start of the `CRep` algorithm
print(f'\nTime elapsed: {np.round(time.time() - time_start, 2)} seconds.')
```

Done! The model has been run and the results are stored into the variable `inferred_parameters` and saved into the file `theta_CRep.npz` in the output folder.


## Analyzing the results
We can now retrieve the results from the saved file. 

```python
filename = conf['out_folder'] + '/theta_' + algorithm + '.npz'
# Load the contents of the file into a dictionary
theta = np.load(filename)
```

```python
# Unpack the latent variables from the results of the `CRep` model
# The `u` variable represents the out-going memberships of the 
# nodes in the graph.
# The `v` variable represents the in-coming memberships of the 
# nodes in the graph.
# The `w` variable represents the affinity of the communities
# The `eta` variable represents the reciprocity coefficient
u, v, w, eta = theta['u'], theta['v'], theta['w'], theta['eta']
```

At this point, one could compare the inferred parameters with the ones used to generate the synthetic network. 

Notice that we don't expect good results in retrieving the communities because we are in a regime with high reciprocity. Indeed, as explained in reference [1], the `CRep` algorithm gives increasingly less weight to the communities as reciprocity increases, resulting in poor community detection performance when the communities are not fully determining edge formation. On the other hand, it well summarizes the network reciprocity.

```python
import networkx as nx

# Print the actual network reciprocity
print(f'Actual network reciprocity: {np.round(nx.reciprocity(A[0]),3)}')
# Print the inferred reciprocity coefficient
print(f'Inferred reciprocity coefficient: {np.round(eta, 3)}')
```

For a more graphical approach on how to investigate the results, we invite the reader to see the tutorial on the 
[`JointCRep` algorithm](./JointCRep.ipynb), where we show how to plot the communities in the network. 

Notice also that the `GM_reciprocity` class provides various methods to create synthetic data according different generative assumptions, and the `reciprocity_planted_network` method used in this tutorial is just an example.


## Summary

This tutorial provides a guide on using the Probabilistic Generative Models (`pgm`) package to generate synthetic network data. In particular, it uses the `CRep` algorithm, which is a probabilistic generative method designed to model directed networks assuming that communities and reciprocity are the main mechanisms for tie formation. 

The tutorial also shows how to analyze the generated network with `pgm` package, and how to infer the latent variables using the `CRep` algorithm. In addition, it guides the user on how to import the inferred results.

Finally, it concludes by referring the user to other tutorials for additional visualizations and functionalities, and to the publication where `CRep` was presented for more details about the meaning of the latent variables.

## References
[1] Safdari, H., Contisciani, M., & De Bacco, C. (2021). Generative model for reciprocity and community detection in networks, _Phys. Rev. Research_ 3, 023209.
