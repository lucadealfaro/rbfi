# RBFI Pytorch Implementation

We provide here two implementations of RBFI: a simple one for people interested in playing with RBFIs per se, and one that is suitable to experimenting with adversarial training, comparisons with ReLU and sigmoid units, and more.

## Simple RBFI implementation

* **mnist_rbfi_pytorch.py** : uses RBFI networks to classify MNIST images.  You can choose the number of layers, the number of neurons in each layer, the and-orness of each layer, and more.  See the options.
* **rbfi_pytorch.py** : implementation of RBFI neurons.

Sample command line:

    python mnist_rbfi_pytorch.py --layers="64,64,64" --andor="v^v^"

Note above that we give one more component to andor than to layers, because an implicit layer with 10 neurons is automatically added at the "top", corresponding to the 10 digits.
Note also that the _andor_ argument is compulsory, as you need to specify layer types.


## RBFI implementation suited for experiments

This implementation is useful if you want to compare the performance of ReLU, RBFI, sigmoidal networks on MNIST, and experiment with adversarial training):

* **mnist_pytorch.py** : uses sigmoidal, ReLU, or RBFI networks over permutation-invariant MNIST.  The learning has many options, as this is geared to experiment with resistance to adversarial attacks.
* **rbf_pseudoderivative_pytorch.py** : experimental implementation of RBFI, with many options that can be tweaked.

Sample command lines:

    python mnist_pytorch.py --relu --layers="64,64,64" --epochs=10

Trains for 10 epochs a ReLU network with layers of 64, 64, 64, 10 neurons on MNIST.
At the end, tests the performance on the test set.
Two files are produced: one that contains the run parameters, and the testing results, and one that contains the trained network weights.

    python mnist_pytorch.py --layers="64,64,64" --rbf --modinf --epochs=10

Same as above, but uses RBFI units.

Let measure_something.json be the generated file containing the run parameters, and measure_something_0.model.json be the trained network.  You can re-read this trained network, and evaluate its susceptibility to FSGM, I-FSGM, and PGD attacks via:

    python mnist_pytorch.py --model_file="measure_something_0" --rbf --modinf --test_fgsm --test_ifgsm --test_pgd

There are many more options; please refer to the command line options of mnist_pytorch.py for more information.

## Framework files (small shared functionality):

* **nn_linear_pytorch_sensitivity** : wrapper around linear layers in Pytorch, to compute network sensitivity for regularization purposes.  Used mainly to expriment.
* **square_distance_loss.py** : implementation of square distance loss.
* **torch_bounded_parameters.py** : implementation of bounded parameters in pytorch.
* **json_plus.py** : enables efficient serialization of numpy arrays (and more) using json.

## Requirements

* PyTorch (0.4.0 or later)
* Numpy

## Author

Luca de Alfaro (luca@dealfaro.org)

## License

BSD
