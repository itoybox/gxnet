import torch
import torch.autograd
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

def myprint( tag, t ):
	print( tag, t.shape, t )

biases = np.array( [ 1, 1 ] )

filters = torch.tensor( np.arange( 0, 2 * 2 * 3 * 3 * 0.1, 0.1 ).reshape( ( 2, 2, 3, 3, ) ),
		requires_grad=True, dtype=torch.double )
X = torch.tensor( np.arange( 0, 2 * 2 * 4 * 4, 1 ).reshape( ( 2, 2, 4, 4, ) ),
		requires_grad=True, dtype=torch.double )

weight = filters
b = torch.tensor( biases, requires_grad=True, dtype=torch.double )

optimizer = optim.SGD( [ X, weight, b ], lr=0.1)
optimizer.zero_grad()

myprint( "X", X )
myprint( "weight", weight )
myprint( "b", b )

Y = torch.nn.functional.conv2d( X, weight, b )

myprint( "Y", Y )

dY = torch.tensor( np.arange( 0, torch.numel( Y ) * 0.1, 0.1 ).reshape( Y.shape ), dtype=torch.double )
myprint( "dY", dY )

Y.backward( dY )
optimizer.step()

myprint( "dWeight", weight.grad )
myprint( "weight", weight )

myprint( "dBiases", b.grad )
myprint( "biases", b )

myprint( "dX", X.grad )
myprint( "X", X )


