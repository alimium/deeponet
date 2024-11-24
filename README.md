# deeponet
An implementation of the DeepONet model along with some examples.

# DeepONet
Deep Operator Networks (*DeepONets*) are a class of deep neural networks, introduced by [Lu et al.](https://doi.org/10.1038/s42256-021-00302-5) ([arXiv](https://doi.org/10.48550/arXiv.1910.03193)) that are designed to learn various non-linear operators as opposed to functions. This is particularly useful in the context of solving ordinary and partial differential equations (ODEs & PDEs) where the solution is an operator that maps the input to the output.

## Architecture
The DeepONet architecture is composed of two main components:

1. Branch net: A deep neural architecture that trnasforms the input sensors* to an intermediate representation of $p$ dimentions.
1. Trunk net: A deep neural architecture that transforms the the evaluation points to an intermediate representation of $p$ dimentions.

> In the simplest form, `branch net` and `trunk net` can be a simple feedforward neural network, sharing the same configurations.

The output of the `branch net` and `trunk net` consists of the inner product of the intermediate representations, summed with a bias term.

> There exists a variation of the DeepONet architecture where $p$ `branch nets` are vertically stacked and each one produces one of the $p$ dimensions of the intermediate representation. All of the elements are then concatenated to form a $p$ dimensional vector.