"""Simple example that demonstrates how geometries from geomstats can be
used in pymanopt to perform optimization on manifolds. It uses the Riemannian
steepest descent solver.

The example currently requires installing the pymanopt HEAD from git:

    pip install git+ssh://git@github.com/pymanopt/pymanopt.git
"""

import logging
import os

import pymanopt
from pymanopt.manifolds import Euclidean
from pymanopt.optimizers import SteepestDescent

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere


class GeomstatsSphere(Euclidean):
    """A simple adapter class which proxies calls by pymanopt's solvers to
    `Manifold` subclasses to the underlying geomstats `Hypersphere` class.
    """

    def __init__(self, ambient_dimension):
        dim = ambient_dimension
        self._sphere = Hypersphere(dim)
        super().__init__(dim)

    def norm(self, base_point, tangent_vector):
        return self._sphere.metric.norm(tangent_vector, base_point=base_point)

    def inner(self, base_point, tangent_vector_a, tangent_vector_b):
        return self._sphere.metric.inner_product(
            tangent_vector_a, tangent_vector_b, base_point=base_point
        )

    def proj(self, base_point, ambient_vector):
        return self._sphere.to_tangent(ambient_vector, base_point=base_point)

    def retr(self, base_point, tangent_vector):
        """The retraction operator, which maps a tangent vector in the tangent
        space at a specific point back to the manifold by approximating moving
        along a geodesic. Since geomstats's `Hypersphere` class doesn't provide
        a retraction we use the exponential map instead (see also
        https://hal.archives-ouvertes.fr/hal-00651608/document).
        """
        return self._sphere.metric.exp(tangent_vector, base_point=base_point)

    def rand(self):
        return self._sphere.random_uniform()

    def randvec(self, base_point):
        random_point = gs.random.normal(size=self.dim + 1)
        random_tangent_vector = self.proj(base_point, random_point)
        return random_tangent_vector / gs.linalg.norm(random_tangent_vector)

    def zerovec(self, base_point):
        return gs.zeros_like(self.rand())


def estimate_dominant_eigenvector(manifold, square_mat):
    """Returns the dominant eigenvector of the symmetric square matrix A by minimizing
    the Rayleigh quotient -x' * A * x / (x' * x).
    """
    num_rows, num_columns = gs.shape(square_mat)
    if num_rows != num_columns:
        raise ValueError("Matrix must be square.")
    if not gs.allclose(gs.sum(square_mat - gs.transpose(square_mat)), 0.0):
        raise ValueError("Matrix must be symmetric.")

    if manifold.dim != num_rows:
        raise ValueError("Manifold's dimension must be same as rank of square matrix")

    @pymanopt.function.numpy(manifold)
    def cost(vector):
        return -gs.dot(vector, gs.dot(square_mat, vector))

    @pymanopt.function.numpy(manifold)
    def egrad(vector):
        return -2 * gs.dot(square_mat, vector)

    problem = pymanopt.Problem(manifold, cost=cost, euclidean_gradient=egrad)
    solver = SteepestDescent()
    return solver.run(problem).point


if __name__ == "__main__":
    if os.environ.get("GEOMSTATS_BACKEND") != "numpy":
        raise SystemExit("This example currently only supports the numpy backend")

    ambient_dim = 128
    sphere = GeomstatsSphere(ambient_dim)
    mat = gs.random.normal(size=(ambient_dim, ambient_dim))
    mat = 0.5 * (mat + mat.T)

    eigenvalues, eigenvectors = gs.linalg.eig(mat)
    dominant_eigenvector = eigenvectors[:, gs.argmax(eigenvalues)]

    dominant_eigenvector_estimate = estimate_dominant_eigenvector(sphere, mat)
    if gs.sign(dominant_eigenvector[0]) != gs.sign(dominant_eigenvector_estimate[0]):
        dominant_eigenvector_estimate = -dominant_eigenvector_estimate

    logging.info(
        "l2-norm of dominant eigenvector: %s", gs.linalg.norm(dominant_eigenvector)
    )
    logging.info(
        "l2-norm of dominant eigenvector estimate: %s",
        gs.linalg.norm(dominant_eigenvector_estimate),
    )
    error_norm = gs.linalg.norm(dominant_eigenvector - dominant_eigenvector_estimate)
    logging.info("l2-norm of difference vector: %s", error_norm)
    logging.info("solution found: %s", gs.isclose(error_norm, 0.0, atol=1e-3))
