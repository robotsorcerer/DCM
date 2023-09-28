__all__         = ["Interval"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2023, Discrete Cosserat SoRO Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "Microsoft License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__comments__    = "Ripped off: https://github.com/algoim/algoim/blob/master/algoim/interval.hpp"
__date__ 		= "March 15, 2023"
__status__ 		= "Not Completed"


class Interval():
    def __init__(self, N, alpha = 0.0, beta = 0.0, eps=0.0):
        """
            Interval arithmetic using a first-order Taylor series with remainder. A function's range of attainable values is
            evaluated as
                f(x_c + y) = alpha + beta.y + [-eps,eps] 
            where alpha is the value of the function at the centre x_c of an interval [x_c - delta, x_c + delta],
            beta is the n-dimensional gradient of the function evaluated at the centre, y is a placeholder for first-order
            variations, and eps bounds the remainder term.
        
            delta is a static (thread-local) variable and should be appropriately initialised before a sequence of calculations
            involving any Interval<N> objects. Thus, all intervals share the same delta bounds. This is thread-safe, however
            note this technique is not appropriate when using a complicated composition of algorithms (e.g., if a method to
            calculate a function's value internally depends on another method that applies separate interval arithmetic
            calculations).
        
            For a general C^2 function f, a first order Taylor series with remainder can be computed via
                f(alpha + beta.y + h(y)) = f(alpha) + f'(alpha)*(beta.y + h(y)) + 0.5 f''(xi) (beta.y + h(y))^2
            where xi is somewhere in the interval [alpha, alpha + beta.y + h(y)]. Thus, if C bounds |f''| on the same interval,
            then one can deduce
                f(alpha + beta.y + h(y)) = f(alpha) + f'(alpha)*beta.y + hh(y)
            where 
                hh(y) = f'(alpha) h(y) + 0.5 f''(xi) (beta.y + h(y))^2
            and
                |hh(y)| <= |f'(alpha)| eps + 0.5 C (|beta|.delta + eps)^2.
        """

        self.alpha = alpha
        self.eps   = eps
        self.N     = N
        self.beta  = [0.0]*self.N

    def max_deviation(self):
        b = self.eps

        for dim in range(0, self.N):
            b += abs(self.beta[dim]) * self.delta[dim]



