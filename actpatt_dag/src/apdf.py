import numpy as np
import pandas as pd


def create(n_layers, cust_ind=None):

    # Layers indexes
    hidden_layers_index = ["l" + str(i) for i in range(n_layers)]

    # Standard indexes
    standard_indexes = ["instance", "label", "output"]

    # Add other indexes if present
    if cust_ind is not None:
        standard_indexes = standard_indexes + cust_ind

    columns = standard_indexes + hidden_layers_index

    # Define a pandas of
    # [instance, label, output, ..., l1, l2, ...]
    df = pd.DataFrame(columns=columns)

    return df


def append(apdf, instances, outputs, labels, hooks):
    """Add instances to pattern data frame
    ...

    Parameters
    ----------
    apdf
    instances
    outputs
    labels
    hooks
    """

    activations = []

    # Get activations from the hook of each layer
    for i, hook in enumerate(hooks):

        act_pattern = hook.get_act()

        activations.append(act_pattern)

    final = list([instances, labels, outputs])
    final.extend(activations)

    final = np.vstack(final)

    # From [[layer 1], ..., [layer n]] to [[instance 1], ..., [instance m]]
    final = final.transpose()

    new_rows = pd.DataFrame(final, columns=apdf.columns)

    return apdf.append(new_rows)


def stats(df_patt, n_layers):
    """Add instances to pattern data frame
    ...

    Parameters
    ----------
    df_patt
    n_layers
    """

    stats = {}

    stats["in-degrees"] = []
    stats["out-degrees"] = []
    stats["n_occurrences"] = []
    stats["reachable"] = []

    # Loop on layers to compute calculations
    for i in range(n_layers):

        # Exact column of layer
        column = 2 + i

        # Group by that column
        df_by_l = df_patt.groupby(df_patt.columns[column])

        # Count predecessors in the DAG, i.e. unique values
        # in the previous col
        in_degrees = df_by_l[df_patt.columns[column - 1]].nunique().values
        stats["in-degrees"].append(in_degrees)

        # Count successors in the DAG, i.e. unique values
        # in the following col
        out_degrees = df_by_l[df_patt.columns[column + 1]].nunique().values
        stats["out-degrees"].append(out_degrees)

        n_occ = df_by_l[df_patt.columns[column - 1]].count().values
        stats["n_occurrences"].append(n_occ)

        # Count number of reachable labels
        reach_out = df_by_l['n_out'].nunique().values
        stats["reachable"].append(reach_out)

    return stats
