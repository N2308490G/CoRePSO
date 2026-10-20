"""Budget, bound mapping and objective-interface checks, using the CPU only."""
import unittest

import numpy as np
import torch
from corepso import minimize


class InterfaceTests(unittest.TestCase):
    def test_budget_bounds_and_reported_best(self):
        torch.set_num_threads(1)
        lower, upper = np.array([-2., 1., -4.]), np.array([3., 6., -1.])
        center = np.array([0.7, 2.2, -2.3])
        batches, values = [], []

        def objective(x):
            batches.append(x.numpy().copy())
            y = ((x - x.new_tensor(center)).square() *
                 x.new_tensor([1., 3., 7.])).sum(-1)
            values.extend(y.tolist())
            return y

        previous = torch.get_default_dtype()
        result = minimize(objective, lower, upper, max_fes=4057,
                          seed=17, lineages=4, particles=8)
        candidates = np.concatenate(batches)
        self.assertEqual(len(candidates), 4057)
        self.assertEqual(result.nfev, 4057)
        self.assertTrue(np.all(candidates >= lower - 1e-12))
        self.assertTrue(np.all(candidates <= upper + 1e-12))
        self.assertAlmostEqual(result.fun, min(values), places=12)
        self.assertAlmostEqual(result.fun, np.sum((result.x-center)**2 * [1, 3, 7]),
                               places=12)
        self.assertTrue(np.all(np.diff(result.history[:, 1]) <= 0))
        self.assertEqual(torch.get_default_dtype(), previous)

    def test_small_budget_and_invalid_objective(self):
        result = minimize(lambda x: x.square().sum(-1), [-2., -1.], [3., 4.],
                          max_fes=7, lineages=4, particles=8)
        self.assertEqual(result.nfev, 7)
        with self.assertRaisesRegex(ValueError, "shape"):
            minimize(lambda x: x.sum(), [-1.], [1.], max_fes=8, lineages=4, particles=2)
        with self.assertRaises(FloatingPointError):
            minimize(lambda x: x[:, 0] * float("nan"), [-1.], [1.],
                     max_fes=8, lineages=4, particles=2)


if __name__ == "__main__":
    unittest.main()
