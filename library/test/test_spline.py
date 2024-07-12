import random
import unittest
import numpy as np

from tplcpp import PolyCubic, PolyQuintic, PolySeptic
from scipy.interpolate import BPoly


class TestSplines(unittest.TestCase):

    def setUp(self):

        random.seed(5454754987)

    def test_cubicSpline(self):

        x0 = random.uniform(-100, 100)
        x1 = random.uniform(-100, 100)

        x0, x1 = min(x0, x1), max(x0, x1)

        s0 = random.uniform(-100, 100)
        s1 = random.uniform(-100, 100)

        v0 = random.uniform(-100, 100)
        v1 = random.uniform(-100, 100)

        p_gt = BPoly.from_derivatives([x0, x1], [[s0, v0], [s1, v1]])
        p = PolyCubic(x0, s0, v0, x1, s1, v1)

        xs = np.linspace(x0, x1, 100)
        for x in xs:
            self.assertAlmostEqual(p.f(x), p_gt(x), places=7)
            self.assertAlmostEqual(p.df(x), p_gt.derivative()(x), places=7)
            self.assertAlmostEqual(p.ddf(x), p_gt.derivative(2)(x), places=7)
            self.assertAlmostEqual(p.dddf(x), p_gt.derivative(3)(x), places=7)

    def test_quinticSpline(self):

        x0 = random.uniform(-100, 100)
        x1 = random.uniform(-100, 100)

        x0, x1 = min(x0, x1), max(x0, x1)

        s0 = random.uniform(-100, 100)
        s1 = random.uniform(-100, 100)

        v0 = random.uniform(-100, 100)
        v1 = random.uniform(-100, 100)

        a0 = random.uniform(-100, 100)
        a1 = random.uniform(-100, 100)

        p_gt = BPoly.from_derivatives([x0, x1], [[s0, v0, a0], [s1, v1, a1]])
        p = PolyQuintic(x0, s0, v0, a0, x1, s1, v1, a1)

        xs = np.linspace(x0, x1, 100)
        for x in xs:
            self.assertAlmostEqual(p.f(x), p_gt(x), places=7)
            self.assertAlmostEqual(p.df(x), p_gt.derivative()(x), places=7)
            self.assertAlmostEqual(p.ddf(x), p_gt.derivative(2)(x), places=7)
            self.assertAlmostEqual(p.dddf(x), p_gt.derivative(3)(x), places=7)

    def test_septicSpline(self):

        x0 = random.uniform(-100, 100)
        x1 = random.uniform(-100, 100)

        x0, x1 = min(x0, x1), max(x0, x1)

        s0 = random.uniform(-100, 100)
        s1 = random.uniform(-100, 100)

        v0 = random.uniform(-100, 100)
        v1 = random.uniform(-100, 100)

        a0 = random.uniform(-100, 100)
        a1 = random.uniform(-100, 100)

        j0 = random.uniform(-100, 100)
        j1 = random.uniform(-100, 100)

        p_gt = BPoly.from_derivatives([x0, x1], [[s0, v0, a0, j0], [s1, v1, a1, j1]])
        p = PolySeptic(x0, s0, v0, a0, j0, x1, s1, v1, a1, j1)

        xs = np.linspace(x0, x1, 100)
        for x in xs:
            self.assertAlmostEqual(p.f(x), p_gt(x), places=7)
            self.assertAlmostEqual(p.df(x), p_gt.derivative()(x), places=7)
            self.assertAlmostEqual(p.ddf(x), p_gt.derivative(2)(x), places=7)
            self.assertAlmostEqual(p.dddf(x), p_gt.derivative(3)(x), places=7)
