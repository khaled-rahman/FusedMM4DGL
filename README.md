## FusedMM in DGL Framework

### Installation
To compile and install, follow the standard guidelines of DGL available at `https://docs.dgl.ai/install/index.html`. A sample file is also included in the link `examples/pytorch/gcn/install.sh`.

### Run GCN using FusedMM
```
$ cd examples/pytorch/gcn
$ python3 train.py --n-epochs 100 --gcn2
```

### Run GCN using DGL's Kernel
```
$ cd examples/pytorch/gcn
$ python3 train.py --n-epochs 100
```

