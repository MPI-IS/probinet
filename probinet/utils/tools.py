"""
This module contains utility functions for data manipulation and file I/O.
"""

import logging
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import networkx as nx
import numpy as np
import pandas as pd
from sparse import COO

from ..models.constants import ATOL_DEFAULT, RTOL_DEFAULT
from ..types import ArraySequence


def can_cast_to_int(string: Union[int, float, str]) -> bool:
    """
    Verify if one object can be converted to integer object.

    Parameters
    ----------
    string : int or float or str
             Name of the node.

    Returns
    -------
    bool : bool
           If True, the input can be converted to integer object.
    """

    try:
        int(string)
        return True
    except ValueError:
        logging.error("Cannot cast %s to integer.", string)
        return False


def is_sparse(X: np.ndarray) -> bool:
    """
    Check whether the input tensor is sparse.
    It implements a heuristic definition of sparsity. A tensor is considered sparse if:
    given
    M = number of modes
    S = number of entries
    I = number of non-zero entries
    then
    N > M(I + 1)

    Parameters
    ----------
    X : ndarray
        Input data.

    Returns
    -------
    Boolean flag: true if the input tensor is sparse, false otherwise.
    """

    # Get the number of dimensions of the input tensor X.
    M = X.ndim

    # Get the total number of elements in the tensor X.
    S = X.size

    # Get the number of non-zero elements in the tensor X using the first
    # dimension of the non-zero indices.
    I = X.nonzero()[0].size

    # Check if the tensor X is sparse based on a heuristic definition of sparsity.
    # A tensor is considered sparse if the total number of elements is greater
    # than (number of non-zero elements + 1) times the number of dimensions.
    return S > (I + 1) * M


def sptensor_from_dense_array(X: np.ndarray) -> COO:
    """
    Create a sparse tensor from a dense array using sparse.COO.

    Parameters
    ----------
    X : ndarray
        Input data.

    Returns
    -------
    COO
        Sparse tensor created from the dense array.
    """
    # Get the non-zero indices and values from the dense array
    coords = np.array(X.nonzero())
    data = X[tuple(coords)].astype(float)

    # Create the sparse tensor using COO format
    sparse_X = COO(coords, data, shape=X.shape)
    return sparse_X


def get_item_array_from_subs(A: np.ndarray, ref_subs: ArraySequence) -> np.ndarray:
    """
    Retrieves the values of specific entries in a dense tensor.
    Output is a 1-d array with dimension = number of non-zero entries.

    Parameters
    ----------
    A : np.ndarray
        The input tensor from which values are to be retrieved.

    ref_subs : Tuple[np.ndarray]
        A tuple containing arrays of indices. Each array in the tuple corresponds to indices along
        one dimension of the tensor.

    Returns
    -------
    np.ndarray
        A 1-dimensional array containing the values of the tensor at the specified indices.
    """
    return np.array([A[tuple(sub)] for sub in zip(*ref_subs)])


def check_symmetric(
    a: Union[np.ndarray, List[np.ndarray]],
    rtol: float = RTOL_DEFAULT,
    atol: float = ATOL_DEFAULT,
) -> bool:
    """
    Check if a matrix or a list of matrices is symmetric.

    Parameters
    ----------
    a : ndarray or list
        Input data.
    rtol : float
           Relative convergence_tol.
    atol : float
           Absolute convergence_tol.

    Returns
    -------
    True if the matrix is symmetric, False otherwise.
    """

    if isinstance(a, list):
        return all(np.allclose(mat, mat.T, rtol=rtol, atol=atol) for mat in a)

    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def build_edgelist(A: COO, layer: int) -> pd.DataFrame:
    """
    Build the edgelist for a given layer of an adjacency tensor in DataFrame format.

    Parameters
    ----------
    A : coo_matrix
        Sparse matrix in COOrdinate format representing the adjacency tensor.
    layer : int
        Index of the layer for which the edgelist is to be built.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the edgelist for the specified layer with columns 'source', 'target', and 'L<layer>'.
    """

    # Convert the input sparse matrix A to COOrdinate format
    A_coo = A.tocoo()

    # Create a dictionary with 'source', 'target', and 'L' keys
    # 'source' and 'target' represent the row and column indices of non-zero elements in A
    # 'L' represents the data of non-zero elements in A
    data_dict = {"source": A_coo.row, "target": A_coo.col, "L" + str(layer): A_coo.data}

    # Convert the dictionary to a pandas DataFrame
    df_res = pd.DataFrame(data_dict)

    return df_res


def output_adjacency(A: List, out_folder: PathLike, label: str):
    """
    Save the adjacency tensor to a file.
    Default format is space-separated .csv with L+2 columns: source_node target_node
    edge_l0 ... edge_lL .

    Parameters
    ----------
    A : ndarray
        Adjacency tensor.
    out_folder : str
        Output folder.
    label : str
        Label of the evaluation file.
    """

    # Concatenate the label with '.dat' to form the evaluation file name
    outfile = label + ".dat"

    # Create a Path object for the evaluation folder
    out_folder_path = Path(out_folder) if isinstance(out_folder, str) else out_folder

    # Create the evaluation directory. If the directory already exists, no exception is raised.
    # parents=True ensures that any missing parent directories are also created.
    out_folder_path.mkdir(parents=True, exist_ok=True)

    # For each layer in A, build an edge list and store them in a list
    df_list = [build_edgelist(A[layer], layer) for layer in range(len(A))]

    # Concatenate all the DataFrames in the list into a single DataFrame
    df = pd.concat(df_list)

    # Save the DataFrame to a CSV file in the evaluation directory
    # index=False prevents pandas from writing row indices in the CSV file
    # sep=' ' specifies that the fields are separated by a space
    df.to_csv(out_folder_path / outfile, index=False, sep=" ")

    # Print the location where the adjacency matrix is saved
    logging.info("Adjacency matrix saved in: %s", out_folder_path / outfile)


def write_adjacency(
    G: List[nx.MultiDiGraph],
    folder: str = "./",
    fname: str = "multilayer_network.csv",
    ego: str = "source",
    alter: str = "target",
):
    """
    Save the adjacency tensor to file.

    Parameters
    ----------
    G : list
        List of MultiDiGraph NetworkX objects.
    folder : str
             Path of the folder where to save the files.
    fname : str
            Name of the adjacency tensor file.
    ego : str
          Name of the column to consider as source of the edge.
    alter : str
            Name of the column to consider as target of the edge.
    """

    N = G[0].number_of_nodes()
    L = len(G)
    B = np.empty(shape=[len(G), N, N])
    for layer in range(len(G)):
        B[layer, :, :] = nx.to_numpy_array(G[layer], weight="weight")
    df_list = []
    for i in range(N):
        for j in range(N):
            Z = 0
            for layer in range(L):
                Z += B[layer][i][j]
            if Z > 0:
                data = [i, j]
                data.extend([int(B[a][i][j]) for a in range(L)])
                df_list.append(data)
    cols = [ego, alter]
    cols.extend(["L" + str(l) for l in range(1, L + 1)])
    df = pd.DataFrame(df_list, columns=cols)
    df.to_csv(path_or_buf=folder + fname, index=False)
    logging.info("Adjacency tensor saved in: %s", folder + fname)


def create_design_matrix(
    metadata: Dict[str, str], nodeID: str = "Name", attr_name: str = "Metadata"
) -> pd.DataFrame:
    """
    Create the design matrix DataFrame from metadata.

    Parameters
    ----------
    metadata : dict
               Dictionary where the keys are the node labels and the values are the metadata associated to them.
    nodeID : str
             Name of the column with the node labels.
    attr_name : str
                Name of the column to consider as attribute.

    Returns
    -------
    X : DataFrame
        Design matrix
    """
    # Create a DataFrame from the metadata dictionary
    X = pd.DataFrame.from_dict(metadata, orient="index", columns=[attr_name])

    # Create a new column with the node labels
    X[nodeID] = X.index

    # Select the columns in the order specified by nodeID and attr_name
    X = X.loc[:, [nodeID, attr_name]]

    return X


def save_design_matrix(
    X: pd.DataFrame, perc: float, folder: str = "./", fname: str = "X_"
):
    """
    Save the design matrix to file.

    Parameters
    ----------
    X : DataFrame
        Design matrix.
    perc : float
           Fraction of match between communities and metadata.
    folder : str
             Path of the folder where to save the files.
    fname : str
            Name of the design matrix file.
    """
    # Construct the file path using f-string formatting and Path
    file_path = Path(folder) / f"{fname}{str(perc)[0]}_{str(perc)[2]}.csv"

    # Save the DataFrame to a CSV file
    X.to_csv(path_or_buf=file_path, index=False)

    # Log the location where the design matrix is saved
    logging.debug("Design matrix saved in: %s", file_path)


def write_design_matrix(
    metadata: Dict[str, str],
    perc: float,
    folder: str = "./",
    fname: str = "X_",
    nodeID: str = "Name",
    attr_name: str = "Metadata",
) -> pd.DataFrame:
    """
    Save the design matrix to file.

    Parameters
    ----------
    metadata : dict
               Dictionary where the keys are the node labels and the values are the metadata associated to them.
    perc : float
           Fraction of match between communities and metadata.
    folder : str
             Path of the folder where to save the files.
    fname : str
            Name of the design matrix file.
    nodeID : str
             Name of the column with the node labels.
    attr_name : str
                Name of the column to consider as attribute.

    Returns
    -------
    X : DataFrame
        Design matrix
    """
    # Create the design matrix DataFrame from metadata
    X = create_design_matrix(metadata, nodeID=nodeID, attr_name=attr_name)

    # Save the design matrix to a CSV file
    save_design_matrix(X, perc=perc, folder=folder, fname=fname)

    return X


def log_and_raise_error(error_type: Type[BaseException], message: str) -> None:
    """
    Logs an error message and raises an exception of the specified type.

    Parameters
    ----------
    error_type : Type[BaseException]
        The type of the exception to be raised.
    message : str
        The error message to be logged and included in the exception.

    Raises
    ------
    BaseException
        An exception of the specified type with the given message.
    """

    # Log the error message
    logging.error(message)

    # Raise the exception
    raise error_type(message)


def flt(x: float, d: int = 3) -> float:
    """
    Round a number to a specified number of decimal places.

    Parameters
    ----------
    x : float
        Number to be rounded.
    d : int
        Number of decimal places to round to.
    Returns
    -------
    float
        The input number rounded to the specified number of decimal places.
    """
    return round(x, d)


def get_or_create_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    """
    Set the random seed and initialize the random number generator.

    Parameters
    ----------
    rng : Optional[np.random.Generator]
        Random number generator. If None, a new generator is created using the seed.

    Returns
    -------
    np.random.Generator
        Initialized random number generator.
    """
    return rng if rng else np.random.default_rng()
