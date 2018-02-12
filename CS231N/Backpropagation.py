# example 3

# f(x,y) = (x+sig(y)) / (sig(x) + (x+y)**2)
# sig(x) = 1 / (1+exp(-x))

import numpy as np

def example(x, y):

	# forward pass
	sigy = 1.0 / (1 + np.exp(-y))
	num =  x + sigy
	sigx = 1.0 / (1 + np.exp(-x))
	xpy = x + y
	xpysqr = xpy ** 2
	den = sigx + xpysqr
	invden = 1 / den
	fxy = num * invden

	# backwards
	dinvden = num
	dden = dinvden * (1 / (den)**2)
	dxpysqr = dden
	dxpy = dxpysqr * 2 * xpy
	dsigx = dden
	dnum = invden
	dsigy = dnum 
	dx = dsigx * (1 - sigx) * sigx
	dy = dsigy * (1 - sigy) * sigy
	dx += 1 * dxpy
	dy += 1 * dxpy
	dx += ddnum
    