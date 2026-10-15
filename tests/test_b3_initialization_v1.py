"""CPU contracts for the initialization-only experiment, not GPU admission."""
import copy
from pathlib import Path
import tempfile
import unittest
import torch
from qvla.extended_peft import zero_state, attach_state
from qvla.b3_initialization_v1 import inherited_cache_contract, INITIALIZATIONS


class InitializationContracts(unittest.TestCase):
    def test_zero_output_preserves_rng_and_is_deterministic(self):
        layer = torch.nn.Linear(16, 12, bias=False)
        before = torch.get_rng_state().clone()
        first = zero_state(layer, "test.module")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        second = zero_state(layer, "test.module")
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        self.assertGreater(torch.count_nonzero(first[0]).item(), 0)
        self.assertEqual(torch.count_nonzero(first[1]).item(), 0)

    def test_zero_branch_learns_and_roundtrips(self):
        model = torch.nn.Module()
        model.add_module("target", torch.nn.Linear(16, 12, bias=False))
        model.requires_grad_(False)
        a, b = zero_state(model.target, "target")
        state = {"target.awq_recovery_lora.0.weight": a,
                 "target.awq_recovery_lora.1.weight": b}
        x = torch.randn(4, 16)
        ref = model.target(x).detach()
        frozen = model.target.weight.detach().clone()
        attach_state(model, state, {"target"})
        self.assertTrue(torch.equal(ref, model.target(x)))
        branch = model.target.awq_recovery_lora
        optimizer = torch.optim.SGD(branch.parameters(), lr=.1)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = (model.target(x) - 1).square().mean()
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters()))
            if step == 0:
                self.assertEqual(torch.count_nonzero(branch[0].weight.grad).item(), 0)
                self.assertGreater(torch.count_nonzero(branch[1].weight.grad).item(), 0)
            else:
                self.assertGreater(torch.count_nonzero(branch[0].weight.grad).item(), 0)
            optimizer.step()
        self.assertTrue(torch.equal(frozen, model.target.weight))
        self.assertFalse(torch.equal(ref, model.target(x)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pt"
            torch.save(branch.state_dict(), path)
            saved = model.target(x).detach().clone()
            with torch.no_grad():
                branch[1].weight.zero_()
            branch.load_state_dict(torch.load(path, weights_only=True))
            self.assertTrue(torch.equal(saved, model.target(x)))

    def test_new_namespace_and_required_inherited_cache(self):
        self.assertEqual(INITIALIZATIONS, ("standard-zero", "response-svd"))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                inherited_cache_contract({"shared_b3_smoke_dir": tmp})


if __name__ == "__main__":
    unittest.main()
