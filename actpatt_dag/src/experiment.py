from actpatt_dag.src.training import train
from actpatt_dag.src.act_recorder import register_hooks
from actpatt_dag.src.create_model import ReLU_net
from actpatt_dag.src.build_apdf import build_apdf
from actpatt_dag.src.stats import Stats
from actpatt_dag.src.early_stopping import EarlyStopping

import os
import torch
import torch.optim as optim
import torch.nn as nn


def experiment(experiment_name,
               trainset,
               testset,
               lr,
               input_size,
               output_size,
               stopping_crit,
               **params):

    print("\n" + experiment_name)

    hidd_size = params.get('hidden_sizes')

    # Create model
    model = ReLU_net(input_size, output_size, hidd_size)

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Get dataloaders
    trainloader = trainset.get_dataloader(64)
    testloader = testset.get_dataloader(8)

    # Register hooks to save activation patterns
    hooks = register_hooks(model)

    # Init the early_stopping class if needed
    if stopping_crit == "early_stopping":
        valset = params.get('valset')
        valset = valset.get_dataloader(8)
        patience = params.get('patience')

        early_stop = EarlyStopping(model, valset, patience, criterion)

        params.update({'early_stop': early_stop})

    # Stats
    stats = Stats()

    stats.add("forgetting_events", {})

    parameters = {
        "data_loader": trainloader,
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "stopping_crit": stopping_crit,
        "hooks": hooks,
        "stats": stats,
    }
    params.update(parameters)

    print("Starting training...")
    model = train(**params)

    print("Building final training and test apdf")
    patt_train_df = build_apdf(model, trainloader, hooks, len(hooks))
    patt_test_df = build_apdf(model, testloader, hooks, len(hooks))

    stats.add("train_apdf", patt_train_df)
    stats.add("test_apdf", patt_test_df)

    cwd = os.getcwd()
    #torch.save(model, cwd + "/data/LOD/MNIST_model_" + experiment_name + ".pkl")

    stats.save(experiment_name + "stats.pkl")

    return stats


def relu_counter(model):
    ReLUs = [module for module in model.modules()
             if type(module) == nn.ReLU]
    return len(ReLUs)
