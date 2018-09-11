# RBFI Pytorch Implementation

The files contained here are:

Simple RBFI usage:

* **mnist_rbfi_pytorch.py** : uses RBFI networks to classify MNIST images.  You can choose the number of layers, the number of neurons in each layer, the and-orness of each layer, and more.  See the options. 
* **rbfi_pytorch.py** : implementation of RBFI neurons. 

Experimental usage (useful if you want to compare the performance of RELU, RBFI, sigmoidal networks on MNIST):

* **mnist_pytorch.py** : uses sigmoidal, RELU, or RBFI networks over permutation-invariant MNIST.  The learning has many options, as this is geared to experiment with resistance to adversarial attacks. 
* **rbf_pseudoderivative_pytorch.py** : experimental implementation of RBFI, with many options that can be tweaked. 

Framework files (small shared functionality):

* **nn_linear_pytorch_sensitivity** : wrapper around linear layers in Pytorch, to compute network sensitivity for regularization purposes.  Used mainly to expriment. 
* **square_distance_loss.py** : implementation of square distance loss. 
* **torch_bounded_parameters.py** : implementation of bounded parameters in pytorch. 

## Sample command lines. 

    python mnist_rbfi_pytorch.py --layers="64,64,64" --andor="v^v^"
    
Note above that we give one more component to andor than to layers, because an implicit layer with 10 neurons is automatically added at the "top", corresponding to the 10 digits. 

## Requirements

* PyTorch 
* Numpy

## Author

Luca de Alfaro (luca@dealfaro.org)

## License

BSD
