# Minimum Implementation of Automatic Differentiation for Deep Learning

## Requirment
- Numpy  
- CuPy (Optional)

## How to use

### Basic Usage
You can use automatic differentiation on Node objects (defined in node/node.py) easily in the same way to use python built-in numerical objects or numpy arrays.

Node object is initialized by passing a numpy array (list is not supported). 

~~~python
import node
import numpy as np
x = node.Node(np.array([1,2,3,4,5]))
~~~

And if you want to compute on GPU, pass a cupy array (when node library is imported, it prints on what device computation is processed).

~~~python
import node
import cupy
x = node.Node(cupy.array([1,2,3,4,5]))
~~~

Numerical operations on Node object are defined in node/op.py and can be used in the same way to python built-in numerical objects or numpy arrays. Values Node objects have are accessed through value property.

~~~python
x = node.Node(np.random.randn(3, 3))
y = node.Node(np.random.randn(3, 3))
z = x + y
print(z.value)
>>> array([[ 0.1901759 , -1.3383007 , -0.43394455],
>>>        [-0.3760382 , -0.14090723, -1.5118892 ],
>>>        [ 1.9656961 , -0.37430343, -0.4439283 ]], dtype=float32)
~~~

Basic operations are supported 

- Addition / Subtraction / Multiplication / Division
- Dot product
- Matrix Transformation (Transpose / Reshape / Expand /...) 
- Mean / Summation
- Pow / Log / Sqrt
- Max
- ...

Element-wise activation functions can be used as a method of Node object.

~~~python
x = node.Node(np.random.randn(3, 3))
y = x.sigmoid()
~~~

Basic activations are supported (defined in node/op.py).

- Sigmoid
- Tanh
- Softmax
- ReLU
- LeakyReLU
- SeLU (refer to https://arxiv.org/abs/1706.02515)

Loss functions are used in the same way to activations except that they take targets (targets are also Node objects).

~~~python
x = node.Node(np.random.randn(1, 2)) # Output (Batch x Class)
y = node.Node(np.array([0, 1])) # Target (1-Of-K Encoded)
z = x.softmax_with_binary_cross_entropy(y)
~~~

Basic loss functions are supported.

- Binary cross entropy
- Softmax with binary cross entropy
- Mean squared error 

Layers which are interpreted as a fixed set of basic operations can be used (layers are defined in node/layers.py).

~~~python
num_in_units = 2 # the number input units
num_h_units = 3 # the number of hidden units

layer = node.Linear(num_in_units, num_h_units)

x = node.Node(np.random.randn(3, num_in_units))
y = layer(x)

print(y.value)
>>> array([[ 0.72083557, -1.6690434 ,  0.36585453],
>>>        [-2.0229297 ,  1.44547   , -2.1265292 ],
>>>        [ 2.1440656 , -2.8574548 ,  1.8037455 ]], dtype=float32)
~~~

Basic layers are supported.

- Linear (or also called Dense in other libraries)
- Convolution / transposed convolution
- Max pooling 
- Batch normalization

### Model Definition

Model is defined in the below way.

~~~python
class Classifier(node.Network):

    def __init__(self, 
                 num_in_units, 
                 num_h_units,
                 num_out_units):
        self.layers = [
            node.Linear(num_in_units, num_h_units),
            node.Linear(num_h_units, num_out_units)
        ]

    def __call__(self, input):
        hidden = self.layers[0](input).tanh()
        output = self.layers[1](hidden)

        return output

classifier = Classifier(10, 50, 10)

x = node.Node(np.random.randn(1, 10))
y = classifier(x)
~~~

**self.layers** property contains layers used in forward computation. 

~~~python
self.layers = [
            node.Linear(num_in_units, num_h_units),
            node.Linear(num_h_units, num_out_units)
        ]
~~~

**\_\_call\_\_** method defines forward computation.

~~~python
def __call__(self, input):
        hidden = self.layers[0](input).tanh()
        output = self.layers[1](hidden)

        return output
~~~

Parameters are accessed through **get_parameters** method of Network class defined in node/network.py. It returns a list of Node objects which contain parameters.

~~~python
print(classifier.get_parameters())
>>> [<node.node.Node at 0x10b8aeba8>,
>>>  <node.node.Node at 0x10b8ae128>,
>>>  <node.node.Node at 0x10b8aeb00>,
>>>  <node.node.Node at 0x10b8aebe0>]
~~~

### Backward Computation

**node/node.py** module has the global variable named **GRAPH** which contains a sequence of pairs of Op objects used and Node objects operations are applied on. Back-propagation is processed by applying **backward** method of Op objects in **GRAPH** backward. After **backward**, gradients w.r.t. each Node objects are contained in **grad** property of Node objects.

~~~python
# At first GRAPH is an empty list.
# GRAPH: 
# []

x = node.Node(np.random.randn(3, 3))
y = node.Node(np.random.randn(3, 3))
z = x + y

# After add operation, a pair of Add Op object and Node object (for z).
# GRAPH: 
# [Pair(op=<node.op.Add object at 0x113b1d710>, node=<node.node.Node object at 0x113b1d4a8>)]

out = z.sigmoid()

# Another pair is added at the end of GRAPH
# GRAPH:
# [Pair(op=<node.op.Add object at 0x113b1d710>, node=<node.node.Node object at 0x113b1d4a8>), 
#  Pair(op=<node.op.Sigmoid object at 0x113b36e10>, node=<node.node.Node object at 0x113b36860>)]

out.backward()

# After backward, GRAPH is flushed
# GRAPH: 
# []
~~~

### Update Parameters

Gradients computed by backward computation are used to update parameters by optimizers in the following way (optimizers are defined in node/optimizer.py). **PARAMETER-LIST** is a list of Node objects which cantains parameters optimized. **LEARNING-RATE** is a scaler to control step size. **KWD** is other arguments which depend on the type of optimizers.

~~~python
# get_parameters method of Network object returns a list of parameters 
optimizer = node.SGD(PARAMETER-LIST, LEARNING-RATE, *KWD)
~~~

**update** method is used to update parameters.

~~~python
optimizer.update()
~~~
