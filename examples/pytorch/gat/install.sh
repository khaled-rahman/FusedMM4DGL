#!/bin/sh

pushd $(pwd)
cd ../../../
module unload gcc
module load gcc
cd build
make clean
cmake ..
make -j32
cd ../python
python3 setup.py install --user
cd ..
popd
python3 train.py --gat2 --dataset cora
