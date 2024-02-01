## Graph Reduced Order Models ##

### Introduction ###

### Train a GNN ###

From root, type

    python network1d/training.py

The parameters of the trained model and hyperparameters will be saved in `models`, in a folder named as the date and time when the training was launched.

### Test a GNN ###

Within the directory `graphs`, type

    python network1d/tester.py $NETWORKPATH

For example,

    python network1d/tester.py models/01.01.1990_00.00.00

This compute errors for all train and test geometries.
In the example, `models/01.01.1990_00.00.00` is a model generated after training (see Train a GNN).

Some already-trained models are included in `models`
