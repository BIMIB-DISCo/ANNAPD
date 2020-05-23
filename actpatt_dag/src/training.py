from __future__ import print_function

import time
import numpy as np
import sys

from tqdm.autonotebook import tqdm
from actpatt_dag.src.build_apdf import build_apdf
from actpatt_dag.src.forget_events import prediction

def train(stopping_crit,
          **params):

    model = params.get('model')

    if stopping_crit == 'early_stopping':
        model = train_validate(**params)
    elif stopping_crit == 'n_epoch':
        train_epochs(**params)
    else:
        raise ValueError(
            'Got unexpected value for stopping_criteria: ' + str(stopping_crit)
        )

    return model

def train_validate(early_stop,
                   **params):

    check_validation = False
    epoch = 1
    while not early_stop.stop():

        time_c = time.time()

        # Epoch step
        running_loss, accuracy = train_epoch(epoch, **params)

        if accuracy > 0.9:
            check_validation = True

        if check_validation:
            early_stop.step()

        print(epoch, accuracy, running_loss, early_stop.min_val_loss,
              early_stop.no_improve, time.time() - time_c)

        # Save statistics
        stats = params.get("stats")
        stats.add("training_loss", running_loss, epoch)
        stats.add("training_accuracy", accuracy, epoch)

        epoch += 1

    return early_stop.get_best_model()


def train_epochs(epochs,
                 **params):
    """
    Train model for the given number of epochs.
    At the end of each epoch call epoch_functions.
    """

    time_c = time.time()

    running_loss = 0

    accuracy = 0

    # Training loop
    for e in tqdm(range(1, epochs + 1)):

        #print_status(e, epochs, time.time() - time_c, running_loss, accuracy)

        # Epoch step
        running_loss, accuracy = train_epoch(e, **params)

        # Save statistics
        stats = params.get("stats")
        stats.add("training_loss", running_loss, e)
        stats.add("training_accuracy", accuracy, e)


def train_epoch(epoch,
                data_loader,
                model,
                optimizer,
                criterion,
                hooks,
                stats,
                **params):
    """
    Train the model for one epoch.
    """
    running_loss = 0
    accuracy = 0

    # Batches loop
    for images, labels, indexes in data_loader:

        # Training pass
        optimizer.zero_grad()

        # Predicted labels
        output = model(images)

        # Compute loss
        loss = criterion(output, labels)

        # Backprop & optimize
        loss.backward()
        optimizer.step()

        # Compute loss
        running_loss += loss.item()

        # Collect predictions
        pred = output.clone().detach().numpy()
        pred = pred.argmax(axis=1)

        labels = labels.numpy()

        # Correct predictions to binary numpy
        corrects = np.where(labels == pred, 1, 0)

        # Compute forgetting events
        prediction(stats.get("forgetting_events"),
                   indexes.numpy(), corrects, epoch)

        # Compute accuracy
        accuracy += sum(corrects) / len(images)


    # Build and save current apdf
    apdf_interval = params.get("apdf_interval")
    if apdf_interval is not None and epoch % apdf_interval == 0:
        print("Building apdf for epoch " + str(epoch))
        apdf = build_apdf(model, data_loader, hooks, len(hooks))
        stats.add("training_apdf", apdf, epoch)

    # Update accuracy
    accuracy = accuracy / len(data_loader)

    return running_loss / len(data_loader), accuracy


def print_status(e, epochs, t, loss, accuracy):
    sys.stdout.write("[%-50s]" % ('=' * int(50 / epochs * (e + 1))))
    sys.stdout.flush()
    sys.stdout.write(" %d%%" % (100 / epochs * e))
    sys.stdout.flush()
    sys.stdout.write(", epoch %d" % (e))
    sys.stdout.flush()
    sys.stdout.write(", loss %f" % (loss))
    sys.stdout.flush()
    sys.stdout.write(", accuracy %f" % (accuracy))
    sys.stdout.flush()

    remain_time = (t) / (e + 1) * (epochs - (e + 1))
    sys.stdout.write(", ETA " +
                     time.strftime('%H:%M:%S', time.gmtime(remain_time)))
    sys.stdout.flush()
    sys.stdout.write('\r')
