from actpatt_dag.src.experiment import experiment
from actpatt_dag.torch_dataset.indexed_dataset import IndexedDataset, random_split
from actpatt_dag.torch_dataset.vision_datasets import get_MNIST
import fire
import torch


def simple(epochs=4, lr=0.001, hs=[32, 32, 32, 32, 32], name=""):

    dataset = get_MNIST(train=True)
    trainset, valset = random_split(dataset, [45000, 5000])
    testset = IndexedDataset(get_MNIST(train=False))

    parameters = {
        "stopping_crit": "n_epoch",
        "epochs": epochs,
        "trainset": trainset,
        "testset": testset,
        "valset": valset,
        "hidden_sizes": hs,
        "input_size": 784,
        "output_size": 10,
        "lr": lr
    }

    experiment("MNIST_" + name, **parameters)


if __name__ == '__main__':
    torch.set_num_threads(4)
    fire.Fire(simple)
