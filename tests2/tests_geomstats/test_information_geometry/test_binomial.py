import random

import pytest

from geomstats.information_geometry.binomial import (
    BinomialDistributions,
    BinomialMetric,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.information_geometry.binomial import (
    BinomialDistributionsTestCase,
    BinomialMetricTestCase,
)
from tests2.tests_geomstats.test_information_geometry.data.binomial import (
    BinomialDistributionsTestData,
    BinomialMetricTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 10),
    ],
)
def spaces(request):
    request.cls.space = BinomialDistributions(n_draws=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestBinomialDistributions(
    BinomialDistributionsTestCase, metaclass=DataBasedParametrizer
):
    testing_data = BinomialDistributionsTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 10),
    ],
)
def equipped_spaces(request):
    space = request.cls.space = BinomialDistributions(
        n_draws=request.param, equip=False
    )
    space.equip_with_metric(BinomialMetric)

    request.cls.data_generator = RandomDataGenerator(space, amplitude=10.0)


@pytest.mark.usefixtures("equipped_spaces")
class TestBinomialMetric(BinomialMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = BinomialMetricTestData()
