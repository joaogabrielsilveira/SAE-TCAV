"""Installed-model compatibility check of native-state extraction on synthetic data."""

import unittest

import numpy as np
import pandas as pd
import torch

from context_comparisons.native import observe_native_states, ordered_states


class NativeStatesTests(unittest.TestCase):
    def test_hook_matches_model_train_embeddings_and_preserves_predictions(self):
        import concept_subgroup_drift_pilot as pilot
        from tabpfn.model.transformer import PerFeatureTransformer

        torch.set_num_threads(1)
        generator = np.random.default_rng(42)
        context = pd.DataFrame(generator.normal(size=(32, 3)).astype(np.float32), columns=["f0", "f1", "f2"])
        context["patient_id"], context["year"], context["y"] = [f"context-{i}" for i in range(32)], 2007, np.arange(32) % 2
        with observe_native_states() as captured:
            system = pilot.fit_frozen_predictor(context, ["f0", "f1", "f2"], "cpu", seed=42, domain_origin=2007)
        initial = ordered_states(system, captured, 32)
        query = context.iloc[:2].drop(columns="y").copy()
        query["patient_id"], query["year"] = ["query-1", "query-2"], 2008
        first = pilot.capture_frozen_predictions(system, query, "cpu")
        second = pilot.capture_frozen_predictions(system, query.iloc[::-1], "cpu")
        np.testing.assert_allclose(first["probabilities"].cpu().numpy(), second["probabilities"].flip(0).cpu().numpy(),
                                   rtol=1e-4, atol=1e-5)
        unobserved = pilot.fit_frozen_predictor(context, ["f0", "f1", "f2"], "cpu", seed=42, domain_origin=2007)
        control = pilot.capture_frozen_predictions(unobserved, query, "cpu")
        np.testing.assert_allclose(first["probabilities"].cpu().numpy(), control["probabilities"].cpu().numpy(),
                                   rtol=1e-4, atol=1e-5)
        with observe_native_states() as fresh_capture:
            fresh = pilot.fit_frozen_predictor(context, ["f0", "f1", "f2"], "cpu", seed=42, domain_origin=2007)
        np.testing.assert_allclose(initial, ordered_states(fresh, fresh_capture, 32), rtol=1e-5, atol=1e-6)
        classifier = system["classifier"]
        exported = []
        def observe_output(module, inputs, output):
            if isinstance(module, PerFeatureTransformer) and isinstance(output, dict):
                exported.append(output["train_embeddings"][:, 0].detach().cpu().numpy().copy())
        handle = torch.nn.modules.module.register_module_forward_hook(observe_output)
        previous = classifier.transformer_predict_kwargs
        try:
            classifier.transformer_predict_kwargs = {**(previous or {}), "get_additional_outputs": ["train_embeddings"]}
            with observe_native_states() as repeated:
                classifier._fit()
        finally:
            classifier.transformer_predict_kwargs = previous
            handle.remove()
        actual = ordered_states(system, repeated, 32)
        self.assertEqual(len(exported), 1)
        np.testing.assert_allclose(actual, exported[0], rtol=1e-5, atol=1e-6)
        self.assertEqual(initial.shape, actual.shape)


if __name__ == "__main__":
    unittest.main()
