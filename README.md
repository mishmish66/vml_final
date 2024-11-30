# vml_final

## Installation

If you want to use accelerated JAX first install that
```
pip install -U jax[cuda12]
```

Once you have jax then install the rest of the package
```
pip install -e .
```

## Usage
To train a network use the `train.ipynb` notebook I've prepared.
It's the most up-to-date file
Still working on verification, but now everything installs and trains.
If you want to figure out some architectural changes or anything else to make the model more accurate that would be awesome.

## Directory structure
The `src/vml_final` directory contains implementation.
- `model.py` contains model implementation of the TCN and the convolutional blocks. This is the file to mess with to change architecture stuff.
- `data.py` contains dataloading stuff. This is the class I'm using to load the csv camargo data.
- `training.py` contains the code that runs training. This takes batches from the dataloader and steps until the dataloader is done indicating that one epoch is complete.