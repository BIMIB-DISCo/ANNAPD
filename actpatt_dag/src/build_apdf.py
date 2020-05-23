import torch

import actpatt_dag.src.apdf as apdf


def build_apdf(model, data, hooks, n_layers):

    # Create empty apdf
    patt_df = apdf.create(n_layers)

    # Dataset loop
    for inputs, labels, indexes in data:

        # Make predictions
        with torch.no_grad():
            output = model(inputs)

        # Compute output
        output = output.clone().detach().numpy()
        output = output.argmax(axis=1)

        labels = labels.numpy()
        indexes = indexes.numpy()

        # Update apdf
        patt_df = apdf.append(patt_df, indexes,
                              output, labels, hooks)

    return patt_df
