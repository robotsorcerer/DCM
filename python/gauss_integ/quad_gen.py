__all__     = ["GaussQuad"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2023, Discrete Cosserat SoRO Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "Microsoft License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__comment__     = "Ripped off: https:#github.com/algoim/algoim/blob/master/algoim/quadrature_general.hpp"
__date__        = "March 15, 2023"


import hashlib

class Node():
    def __init__(self, x, w):
        self.x = x 
        self.w = w 

class QuadratureRule():
    def __init__(self, N):
        """
            nodes: List of Nodes
        """
        # assert isinstance(nodes, list), " nodes must be a list."
        self.nodes = []
        self.N     = N

    def eval_integrand(self, x, w):
        self.nodes.append(Node(x, w))

    def R_operator(self, funcal):
        "Evaluate an integral applied to a given functional"
        summ = 0 

        for pts in self.nodes:
            summ += funcal(pt.x) * pt.w 
        
        return summ

    def sum_weights(self):
        summ = 0 

        for pt in self.nodes:
            summ += pt.w

        return summ 

class Hyperrectangle():
    def __init__(self, mini, maxi):
        """
            Inputs (ints): mini, maxi
                .define the extent of a hyperrectangle, i.e., the set of 
                points x = x(0), ..., x(N-1) such that xmin(i) <= x(i) <= xmax(i) for all i.
        """
        self.rect_range = [mini, maxi]

    def side(self, s):
        return self.rect_range[s]

    def minimum(self):
        return self.rect_range[0]

    def minimum(self, idx):
        return self.rect_range[0][idx]

    def maximum(self):
        return self.rect_range[-1]

    def maximum(self, idx):
        return self.rect_range[-1][idx]

    def extent(self):
        return self.rect_range[-1] - self.rect_range[0]

    def extent(self, idx):
        return self.rect_range[-1][idx] - self.rect_range[0][idx]

    def midpoint(self):
        return (self.rect_range[-1] - self.rect_range[0])*0.5

    def midpoint(self, idx):
        return (self.rect_range[-1][idx] - self.rect_range[0][idx]   ) *0.5

    def __hash__(self):
        return int(hashlib.md5(str(self.rect_range).encode('utf-8')).hexdigest(),16)

    def __eq__(self, other_self):        
        for dim in range(self.N):
            if self.rect_range[0][dim] != other_self.rect_range[0][dim] or self.rect_range[-1][dim] != self.rect_range[-1][dim]:
                return False
        return True 

def quad_gen(phi, rect_range, dim, side, q0, N):
    """
       Main engine for generating high-order accurate quadrature schemes for explicitly defined domains in
       hyperrectangles. The function is given by the function object phi, the hyperrectangle is
       specified by xrange, and qo determines the number of quadrature points in the underlying 1D Gaussian
       quadrature schemes. Specifically,
       - phi is a user-defined function object which evaluates the function.

         In the simplest case, the role of phi is to simply evaluate the 
         function (e.g., return x(0)*x(0) + x(1)*x(1) - 1; for a unit circle). To enable a certain kind of interval arithmetic, similar to
         automatic differentiation. In essence, the interval arithmetic automatically computes a first-order
         Taylor series (with bounded remainder) of the given level set function, and uses that to make
         decisions concerning the existence of the interface and what direction to use when converting the
         implicitly defined geometry into the graph of an implicitly defined height function. This requirement
         on phi being able to correctly perform interval arithmetic places restrictions on the type of level
         set functions quadGen can be applied to.
       - xrange is a user-specified bounding box, indicating the extent of the hyperrectangle in N dimensions
         to which the quadrature algorithm is applied.
       - dim is used to specify the type of quadrature:
          - If dim < 0, compute a volumetric quadrature scheme, whose domain is implicitly defined by
            {phi < 0} intersected with xrange.
          - If dim == N, compute a curved surface quadrature scheme, whose domain is implicitly defined by
            {phi == 0} intersected with xrange.
          - If 0 <= dim && dim < N, compute a flat surface quadrature scheme for one of the sides of the
            hyperrectangle, i.e., {phi < 0}, intersected with xrange, intersected with the face
            {x(dim) == xrange(side)(dim)}.
       - side is used only when 0 <= dim && dim < N and specifies which side of the hyperrectangle to restrict
         to, either side == 0 or side == 1 for the left or right face, respectively (with normal pointing
         in the direction of the dim-th axis).
       - qo specifies the degree of the underlying one-dimensional Gaussian quadrature scheme and must satisfy
         1 <= qo && qo <= 10.
    """

    qrule = QuadratureRule(N)
    free  = [True] * N 

    if (0 <= dim and dim < N):
        # Volume integral for one of the sides of a hyperrectangle (in dimensions N - 1)
        assert side == 0 or side == 1
        # psi[0] = PsiCode<N>(set_component<int,N>(0, dim, side), -1);
        # free(dim) = false;
        Integral<N-1,N,F,QuadratureRule<N>,false>(phi, q, free, psi, 1, xrange, qo)
        # The quadrature method is given a restricted level set function to work with, but does not actually
        # initialise the dim-th component of each quadrature node's position. Do so now.
        for (auto& node : q.nodes)
            node.x(dim) = xrange.side(side)(dim)
    elif (dim == N):
        # Surface integral
        psi[0] = PsiCode<N>(0, -1);
        Integral<N,N,F,QuadratureRule<N>,true>(phi, q, free, psi, 1, xrange, qo);

    else:
        # Volume integral in the full N dimensions
        psi[0] = PsiCode<N>(0, -1);
        Integral<N,N,F,QuadratureRule<N>,false>(phi, q, free, psi, 1, xrange, qo);
    return q;


