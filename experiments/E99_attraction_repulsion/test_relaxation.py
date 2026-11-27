"""Analytic checks of force direction and strict modification budgets."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from .relaxation import Config, _pairs, relax
from .pipeline import preprocess, solve


class CentralForceChecks(unittest.TestCase):
    def test_inverse_area_response_survives_safety_limits(self):
        # Equal and opposite forces, 4:1 areas. Both weak attraction and strong
        # repulsion must give a 1:4 displacement ratio, including capped steps.
        for gap in (.1, -.2):
            p = np.array([[10., 10., 2., 2.], [12. + gap, 10.5, 1., 1.]])
            out, d = relax(p, config=Config(iterations=1))
            move = out[:, :2] - p[:, :2]
            np.testing.assert_allclose(move[0] * 4, -move[1], atol=2e-14)
            self.assertGreater(np.linalg.norm(move[1]), np.linalg.norm(move[0]))
            self.assertLessEqual(d['trace'][0]['max_step_fraction'], .02 + 1e-12)
        p = np.array([[10., 10., 2., 2.], [11.8, 10.5, 1., 1.]])
        out, _ = relax(p, config=Config(iterations=8))
        np.testing.assert_allclose(((out[:,:2]-p[:,:2])*np.array([4.,1.])[:,None]).sum(0), 0., atol=3e-14)

    def test_attraction_contact_and_cutoff(self):
        for gap, sign in ((-.2, -1), (0., 0), (.2, 1), (1., 0), (100., 0)):
            p = np.array([[5., 5., 2., 2.], [7. + gap, 5., 2., 2.]])
            i, j, u, g, f = _pairs(p, Config())
            self.assertAlmostEqual(g[0], gap)
            self.assertEqual(np.sign(f[0]), sign)
            out, _ = relax(p)
            if sign:
                self.assertEqual(np.sign(out[0, 0] - p[0, 0]), sign)
                np.testing.assert_allclose(out[0, :2] - p[0, :2], -(out[1, :2] - p[1, :2]), atol=2e-14)
            else:
                np.testing.assert_array_equal(out, p)

    def test_rectangle_radial_geometry_and_direction(self):
        p = np.array([[10., 10., 10., 1.], [14., 10.7, 1., 3.]])
        _, _, unit, gap, force = _pairs(p, Config())
        self.assertLess(gap[0], 0)
        self.assertLess(force[0], 0)
        center = p[:, :2] + p[:, 2:] / 2
        out, _ = relax(p, config=Config(iterations=1))
        move = out[0, :2] - p[0, :2]
        self.assertAlmostEqual(move[0] * unit[0, 1] - move[1] * unit[0, 0], 0., places=13)
        self.assertLess(np.dot(move, center[1] - center[0]), 0)

    def test_containment_coincidence_and_anchor(self):
        p = np.array([[10., 10., 8., 8.], [13., 13., 2., 2.]])
        out, d = relax(p, [True, False], config=Config(iterations=8), return_history=True)
        np.testing.assert_array_equal(out[0], p[0])
        self.assertGreater(np.linalg.norm(out[1, :2] - p[1, :2]), 0)
        np.testing.assert_array_equal(out, relax(p, [True, False], config=Config(iterations=8))[0])
        self.assertLessEqual(d['max_path_fraction'], .06 + 1e-12)
        np.testing.assert_array_equal(relax(p, [True, True])[0], p)

    def test_synchronous_budget_and_scale(self):
        p = np.array([[10., 10., 3., 2.], [11., 10., 2., 4.], [12., 12., 4., 1.], [9., 11., 2., 3.]])
        original = p.copy()
        out, d = relax(p, config=Config(iterations=8), return_history=True)
        history = d['history']
        steps = np.linalg.norm(np.diff(history[:, :, :2], axis=0), axis=-1)
        local = p[:, 2:].min(1)
        self.assertTrue((steps <= .02 * local + 1e-12).all())
        self.assertTrue((steps.sum(0) <= .06 * local + 1e-12).all())
        np.testing.assert_array_equal(p, original)
        np.testing.assert_array_equal(out[:, 2:], p[:, 2:])
        np.testing.assert_allclose(relax(p * 7, config=Config(iterations=8))[0], out * 7, atol=2e-12)
        moved = p.copy(); moved[:, :2] += 17
        expected = out.copy(); expected[:, :2] += 17
        np.testing.assert_allclose(relax(moved, config=Config(iterations=8))[0], expected, atol=2e-12)
        np.testing.assert_array_equal(relax(p, config=Config(iterations=0))[0], p)

    def test_public_adapter_invalid_inputs_and_file_cli(self):
        p = np.array([[5., 5., 2., 2.], [6.8, 5., 2., 2.]])
        c = np.zeros((2, 5)); c[0, 1] = 1
        s = np.full((2, 4), -1.); s[0] = p[0]
        inputs = (2, np.array([4., 4.]), None, None, None, c, s)
        expected, _ = preprocess(p, inputs)
        called = []
        def planner(draft, public, **kwargs):
            called.append(draft.copy())
            return draft, {'called': True}
        out, trace = solve(p, inputs, planner)
        np.testing.assert_array_equal(called[0], expected)
        self.assertTrue(trace['planner']['called'])
        for kwargs in ({'iterations': 9}, {'iterations': -1}, {'iterations': 1.5}, {'step_fraction': .03}, {'total_fraction': .07}, {'step_size': float('nan')}):
            with self.assertRaises(ValueError): Config(**kwargs)
        for bad in (np.full((2, 4), np.nan), [[0, 0, 0, 1]], [[-1, 0, 1, 1]], []):
            with self.assertRaises(ValueError): relax(bad)
        with self.assertRaises(ValueError): relax(p, [False])
        with self.assertRaises(ValueError): preprocess(p + .01, inputs)
        workspace = Path(__file__).resolve().parents[2]
        cache = (workspace / '.cache/E99_cli_checks').resolve()
        assert cache.is_relative_to(workspace)
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            assert Path(tmp).resolve().is_relative_to(cache)
            import subprocess, sys
            source, destination = Path(tmp)/'input.npz', Path(tmp)/'output.npz'
            np.savez(source, positions=p, preplaced=np.array([True, False]))
            command = [sys.executable, '-B', '-m', 'experiments.E99_attraction_repulsion', '--input', str(source), '--output', str(destination)]
            subprocess.run(command, check=True, capture_output=True)
            with np.load(destination, allow_pickle=False) as result:
                np.testing.assert_array_equal(result['positions'], expected)
                self.assertEqual(len(result['history']), 5)
            denied = subprocess.run(command, capture_output=True)
            self.assertNotEqual(denied.returncode, 0)


if __name__ == '__main__':
    unittest.main()
