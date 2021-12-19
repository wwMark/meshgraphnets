# PyTorch version of Learning Mesh-Based Simulation with Graph Networks (ICLR 2021)

Source repository: https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets

Video site: [sites.google.com/view/meshgraphnets](https://sites.google.com/view/meshgraphnets)

Paper: [arxiv.org/abs/2010.03409](https://arxiv.org/abs/2010.03409)

## Overview

The code in this repository is the PyTorch version of Learning Mesh-Based Simulation with Graph Networks. Currently, the code of **cloth simulation** can be
run on both windows and linux. Development environment is PyCharm 2021.1.3. Package dependencies are defined
in requirements.txt, please install all dependencies before runnning. The result of this version of MGN network is not 100% conform to the original code, please tune it according to your need. Other GNN networks code will be added to this repository in the future to explore the GNN performance for physical simulation.

## New Features compared to original MeshGraphNets

The novel ripple model, inspired by water ripple, is based on deepmind's meshgraphnets and utilizes ripples to enhance the information propagation. Further new features such as different aggregation methods, attention and stochastic message passing are also already added.

## Setup

Install dependencies:

    pip install -r requirements.txt

Download a dataset:

    mkdir -p ${DATA}
    bash meshgraphnets/download_dataset.sh flag_simple ${DATA}

Go to the dataset directory and generate .idx file(needed by package tfrecord for reading .tfrecord file in PyTorch):

    python -m tfrecord.tools.tfrecord2idx <file>.tfrecord <file>.id
    
Configure running goals by setting the variables and flags variables at the beginning of run_model.py, which includes
running mode (training/evaluation), model, epochs, saving path and etc.

## Running the model

Run the code after configurating the code:

    python run_model.py

Plot a trajectory (rollout.pkl path can be defined inside plot_cloth.py):

    python plot_cloth.py

## Datasets

Datasets can be downloaded using the script `download_dataset.sh`. They contain a metadata file describing the available
fields and their shape, and tfrecord datasets for train, valid and test splits. Dataset names match the naming in the
paper. The following datasets are available:

    airfoil
    cylinder_flow
    deforming_plate
    flag_minimal
    flag_simple
    flag_dynamic
    sphere_simple
    sphere_dynamic

`flag_minimal` is a truncated version of flag_simple, and is only used for integration tests.
