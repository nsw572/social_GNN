import unittest

import numpy as np

from social_gnn.data import SocialTrialPackage
from social_gnn.evaluate_role_binding import role_binding_targets


def package(values):
    timesteps = values.shape[0]
    return SocialTrialPackage(
        trial_id="role_test",
        node_features=np.arange(timesteps * 12, dtype=np.float32).reshape(timesteps, 2, 6),
        node_mask=np.ones((timesteps, 2), dtype=bool),
        edge_values=values.astype(np.float32),
        edge_confidence=np.full_like(values, 0.8, dtype=np.float32),
        patch_start_s=np.arange(timesteps, dtype=np.float64) * 0.16,
        patch_end_s=(np.arange(timesteps, dtype=np.float64) + 1) * 0.16,
        identity=np.asarray(["A", "B"]),
    )


class RoleBindingTest(unittest.TestCase):
    def test_targets_are_exactly_invariant_to_animal_permutation(self):
        rng = np.random.default_rng(12)
        original = package(rng.normal(size=(8, 2, 2, 8)))
        order = np.asarray([1, 0])
        swapped = SocialTrialPackage(
            trial_id="role_test_swapped",
            node_features=original.node_features[:, order],
            node_mask=original.node_mask[:, order],
            edge_values=original.edge_values[:, order][:, :, order],
            edge_confidence=original.edge_confidence[:, order][:, :, order],
            patch_start_s=original.patch_start_s,
            patch_end_s=original.patch_end_s,
            identity=original.identity[order],
        )
        first, first_confidence, names = role_binding_targets(original)
        second, second_confidence, second_names = role_binding_targets(swapped)
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(first_confidence, second_confidence)
        self.assertEqual(names, second_names)

    def test_role_target_changes_when_one_cross_trait_assignment_changes(self):
        values = np.zeros((1, 2, 2, 8), dtype=np.float32)
        values[0, 0, 1, 3], values[0, 1, 0, 3] = 1.0, -1.0
        values[0, 0, 1, 5], values[0, 1, 0, 5] = 2.0, -2.0
        first = role_binding_targets(package(values))[0][0, 0]
        values[0, 0, 1, 5], values[0, 1, 0, 5] = -2.0, 2.0
        second = role_binding_targets(package(values))[0][0, 0]
        self.assertEqual(first, -second)


if __name__ == "__main__":
    unittest.main()
