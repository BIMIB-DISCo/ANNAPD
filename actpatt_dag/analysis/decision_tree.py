from scipy.stats import entropy
import numpy as np
import pandas as pd

class DecisionTree:

    def __init__(self, dataframe, col, split_col="output"):

        # Set of clusters that may be splitted
        unchecked_views = [(len(col), dataframe.instance.to_numpy())]
        
        # Final clusters
        clusters = list()
        level = list()
        purity = list()
        correct = list()

        # Perform recursive splitting
        while unchecked_views != []:

            # Extract a parent cluster
            parent = unchecked_views.pop(0)

            # Get the last splitting attribute
            attribute = parent[0]

            # Get the parent view
            current_instances = parent[1]
            current_view = dataframe.loc[dataframe.instance.isin(current_instances)]

            # Consider the next attribute
            attribute = attribute - 1
    
            # Split set
            split = list()
            for att in current_view[col[attribute]].unique():
                new_cluster = current_view.loc[current_view[col[attribute]] == att].instance
                split.append(new_cluster)

            # Compute parent entropy
            A = current_view[split_col]
            _, counts = np.unique(A, return_counts=True)
            parent_entropy = entropy(counts, base=2)

            # Compute weighted split entropy
            child_entropy = 0
            for s in split:

                A = dataframe.loc[dataframe.instance.isin(s)][split_col]
                _, counts = np.unique(A, return_counts=True)
                s_ent = entropy(counts, base=2)

                s_w_ent = s_ent * len(s) / len(current_instances)

                child_entropy = child_entropy + s_w_ent

            # Decide if perform the split or not
            if parent_entropy - child_entropy > 0:
                for s in split:
                    if attribute == 0 or len(s) == 1:
                        clusters.append(s)
                        level.append(attribute)
                        purity.append(dataframe.loc[dataframe.instance.isin(s)]['correct'].sum()/len(s))
                        correct.append(dataframe.loc[dataframe.instance.isin(s)]['correct'])
                    else:
                        unchecked_views.append((attribute, s))
            else:
                clusters.append(current_instances)
                level.append(attribute)
                purity.append(dataframe.loc[dataframe.instance.isin(current_instances)]['correct'].sum()/len(current_instances))
                correct.append(dataframe.loc[dataframe.instance.isin(current_instances)]['correct'])

            df_size = len(dataframe)
            sum = 0
            for s in clusters:
                sum = sum + len(s)
            perc = int(100 / df_size * sum)
            print("Progress: " + str(perc) + "/100", end='\r')
        
        cluster_id = []
        instance = []
        cluster_size = []
        
        for i, c in enumerate(clusters):
            cluster_id.extend([i]*len(c))
            cluster_size.extend([len(c)]*len(c))
            instance.extend(c)
        
        cols = np.array([instance, cluster_size, cluster_id])
        rows = np.transpose(cols)
        clusters_df = pd.DataFrame(rows, columns=['instance', 'clust_size', 'clust_id'])
        
        self.clust_df = clusters_df
        self.clusters = clusters
        self.level = level
        self.purity = purity
        self.correct = correct
        
