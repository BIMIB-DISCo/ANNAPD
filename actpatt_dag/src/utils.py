import numpy as np

from tqdm import tqdm


def rgini(array):
    if len(array) == 0:
        return 0

    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)  # values cannot be negative

    array = np.sort(array)  # values must be sorted
    index = np.arange(1, array.shape[0] + 1)  # index per array element
    n = array.shape[0]  # number of array elements
    # Gini coefficient
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def extract_data(collected_data, hidden_layers_index):

    data_epochs = []

    instances_data = collected_data["Dag"]

    for i in tqdm(range(len(instances_data))):

        data_epochs.append({})
        data = data_epochs[-1]

        df = instances_data[i]["df"]

        df['correct'] = (df['n_out'] == df['out']).astype(int)

        # Groupby each layer + net output --> dendrogram
        grouped_patt = df.groupby(hidden_layers_index)
        correct_patt = grouped_patt['correct'].agg(["count", "sum"], axis=1)

        instance_patt = grouped_patt['instance', 'n_out', 'correct'].agg(
            clusters=('instance', lambda x: list(x)),
            cluster_labels=('n_out', lambda x: list(x)),
            cluster_corrects=('correct', lambda x: list(x)))

        clust_len = list(correct_patt['count'])
        clust_sum = list(correct_patt['sum'])
        clust_ins = [list(instance_patt['clusters']),
                     list(instance_patt['cluster_labels']),
                     list(instance_patt['cluster_corrects'])]

        data["dendr-clusters"] = clust_ins
        data["dendr-clusters-size"] = clust_len
        data["dendr-purity"] = [int(s) / int(l)
                                for s, l in zip(clust_sum, clust_len)]

    return data_epochs
