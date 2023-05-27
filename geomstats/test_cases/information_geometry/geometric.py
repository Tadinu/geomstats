from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)


class GeometricDistributionsTestCase(InformationManifoldMixinTestCase, OpenSetTestCase):
    pass


class GeometricMetricTestCase(RiemannianMetricTestCase):
    pass
