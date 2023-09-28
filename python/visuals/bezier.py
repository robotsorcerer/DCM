__all__ = ["Bezier"]

import numpy as np

class Bezier():

    def TwoPoints(t, P1, P2):
        """
        Returns a point between P1 and P2, parametised by t.
        INPUTS:
            t     float/int; a parameterisation.
            P1    numpy array; a (concatenation of) point(s).
            P2    numpy array; a (concatenation of) point(s).
        OUTPUTS:
            Q1    numpy array; a point.
        """

        if not isinstance(P1, np.ndarray) or not isinstance(P2, np.ndarray):
            raise TypeError('Points must be an instance of the numpy.ndarray!')
        if not isinstance(t, (int, float)):
            raise TypeError('Parameter t must be an int or float!')

        Q1 = (1 - t) * P1 + t * P2
        return Q1

    def linear_interp(t, P0, P1):
        """
            Inputs:
                t: A multidim array of scalar multipliers such that for each t,
                    0 <= t <= 1
                P0: List of points on a left shifted index 
                P1: List of points on a right shifted index 

            Carry out a linear interpolation between time-shifted points
            P0 and P1 such that:

                B(t)  =  P_0 + t P_1 - P_0
                B(t)  = (1-t) P_0 + t P_1,   0 <= t <= 1.
        """

        return (1-t) * P0 + t * P1

    def GenCurve(t_values, points):
        """
        Returns a list of points interpolated by the Bezier process
        INPUTS:
            t_values: List of floats/ints; a parameterisation.   
            points       list of numpy arrays; points.
        OUTPUTS:
            newpoints    list of numpy arrays; points.
        """

        assert len(t_values)==len(points), "For efficiency, the length of the time and point arrays must be same."

        if not hasattr(t_values, '__iter__'):
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")
        if len(t_values) < 1:
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")
        if not isinstance(t_values[0], (int, float)):
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")


        idx1 = np.arange(0, len(points)-1, dtype=np.intp)
        idx2 = np.arange(1, len(points), dtype=np.intp)
        points1 = points[idx1,:]; points1 = np.insert(points1, -1, points[-1], axis=0)
        points2 = points[idx2,:]; points2 = np.insert(points2, -1, points[-1], axis=0)
        
        newpoints = Bezier.linear_interp(t_values, points1, points2)

        return newpoints        