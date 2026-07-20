"""
Smoke test for da3_slam.backend.processing.factor_graph (SL(4) pose graph).

All tests use synthetic poses — no images, models, or GPU required.
Requires a GTSAM build with SL4 support (gtsam.SL4, BetweenFactorSL4).

Tests:
  1. Single frame      — prior pins the pose at its initial value
  2. Chain recovery    — between-factors reproduce a known trajectory
  3. Loop closure      — a loop factor corrects an injected drift
  4. Warm start        — re-optimising an optimised graph converges instantly
  5. Result interface  — pose(), final_error, iterations types and shapes

Usage:
    python scripts/test_pose_graph.py
    python scripts/test_pose_graph.py --verbose   # show GTSAM LM output
"""

import argparse

import numpy as np

from smoke_test_utils import header, check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true",
                        help="Show GTSAM LM optimisation output")
    return parser.parse_args()


# ── synthetic data ────────────────────────────────────────────────────────────

def se3(translation, rotation_z_deg: float = 0.0) -> np.ndarray:
    """(4, 4) SE(3) pose with a rotation about Z.  det = 1, so it is in SL(4)."""
    angle = np.radians(rotation_z_deg)
    c, s = np.cos(angle), np.sin(angle)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


def relative(pose_a: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
    """Between-factor measurement for cam-to-world nodes: w2c_a @ c2w_b."""
    return np.linalg.inv(pose_a) @ pose_b


def default_noise():
    from da3_slam.backend.processing.factor_graph import NoiseConfig
    return NoiseConfig(prior_sigma=1e-6, between_sigma=0.05, loop_sigma=0.05)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_single_frame(verbose: bool):
    from da3_slam.backend.processing.factor_graph import PoseGraph

    header("1 — Single frame: prior pins the pose")

    graph = PoseGraph(default_noise())
    graph.add_frame(0, np.eye(4))
    graph.add_prior(0)

    check("n_nodes = 1", graph.n_nodes == 1)
    check("n_factors = 1 (prior only)", graph.n_factors == 1)

    result = graph.optimize(verbose=verbose)
    check("result has 1 pose", len(result.frame_poses) == 1)
    check("final_error is finite", np.isfinite(result.final_error))
    check("pose(0) ≈ identity", np.allclose(result.pose(0), np.eye(4), atol=1e-4))
    print(f"  cost={result.final_error:.2e}  iters={result.iterations}")


def test_chain_recovery(verbose: bool):
    from da3_slam.backend.processing.factor_graph import PoseGraph

    header("2 — Chain of 4 frames: trajectory recovery")

    truth = [
        se3([0.0, 0.0, 0.0]),
        se3([0.5, 0.0, 0.0], rotation_z_deg=5),
        se3([1.0, 0.1, 0.0], rotation_z_deg=10),
        se3([1.3, 0.05, 0.0], rotation_z_deg=12),
    ]

    graph = PoseGraph(default_noise())
    for seq_idx, pose in enumerate(truth):
        graph.add_frame(seq_idx, pose)
    graph.add_prior(0)
    for i in range(1, len(truth)):
        graph.add_between(i - 1, i, relative(truth[i - 1], truth[i]))

    check("n_nodes = 4", graph.n_nodes == 4)
    check("n_factors = 4 (1 prior + 3 between)", graph.n_factors == 4)

    result = graph.optimize(verbose=verbose)
    for seq_idx, pose in enumerate(truth):
        err = float(np.linalg.norm(result.pose(seq_idx) - pose))
        check(f"pose({seq_idx}) matches ground truth  err={err:.2e}", err < 1e-3)


def test_loop_closure(verbose: bool):
    from da3_slam.backend.processing.factor_graph import PoseGraph

    header("3 — Loop closure corrects injected drift")

    # Ground-truth square path returning to the start
    truth = [
        se3([0.0, 0.0, 0.0]),
        se3([1.0, 0.0, 0.0]),
        se3([1.0, 1.0, 0.0]),
        se3([0.0, 1.0, 0.0]),
        se3([0.0, 0.05, 0.0]),  # back near the start
    ]

    # Odometry with drift: each between-measurement gets a small bias
    drift = se3([0.04, 0.02, 0.0])
    graph = PoseGraph(default_noise())
    pose_est = truth[0]
    graph.add_frame(0, pose_est)
    graph.add_prior(0)
    for i in range(1, len(truth)):
        measured = relative(truth[i - 1], truth[i]) @ drift
        pose_est = pose_est @ measured
        graph.add_frame(i, pose_est)
        graph.add_between(i - 1, i, measured)

    drift_before = float(np.linalg.norm(
        graph.get_pose(len(truth) - 1)[:3, 3] - truth[-1][:3, 3]
    ))

    # Loop closure: drift-free measurement between last and first frame
    graph.add_between(len(truth) - 1, 0, relative(truth[-1], truth[0]), loop=True)
    result = graph.optimize(verbose=verbose)

    drift_after = float(np.linalg.norm(
        result.pose(len(truth) - 1)[:3, 3] - truth[-1][:3, 3]
    ))
    print(f"  end-pose drift: before={drift_before:.4f}m  after={drift_after:.4f}m")
    check("loop closure reduces end-pose drift", drift_after < drift_before)


def test_warm_start(verbose: bool):
    from da3_slam.backend.processing.factor_graph import PoseGraph

    header("4 — Warm start")

    graph = PoseGraph(default_noise())
    pose = np.eye(4)
    graph.add_frame(0, pose)
    graph.add_prior(0)
    rng = np.random.default_rng(42)
    for i in range(1, 8):
        step = se3(rng.normal(scale=0.3, size=3), rotation_z_deg=rng.normal(scale=4))
        prev = pose
        pose = pose @ step
        graph.add_frame(i, pose)
        graph.add_between(i - 1, i, relative(prev, pose))

    r1 = graph.optimize(verbose=verbose)
    r2 = graph.optimize(verbose=verbose)  # warm-started from r1's values

    check("warm-start result is finite", np.isfinite(r2.final_error))
    check("warm-start error ≤ cold-start error",
          r2.final_error <= r1.final_error + 1e-9)
    check("warm-start uses ≤ iterations", r2.iterations <= r1.iterations)
    print(f"  cold: {r1.iterations} iters  cost={r1.final_error:.2e}")
    print(f"  warm: {r2.iterations} iters  cost={r2.final_error:.2e}")


def test_result_interface():
    from da3_slam.backend.processing.factor_graph import PoseGraph

    header("5 — OptimizationResult interface")

    graph = PoseGraph(default_noise())
    truth = [se3([float(i), 0.0, 0.0]) for i in range(3)]
    for seq_idx, pose in enumerate(truth):
        graph.add_frame(seq_idx, pose)
    graph.add_prior(0)
    for i in range(1, 3):
        graph.add_between(i - 1, i, relative(truth[i - 1], truth[i]))
    result = graph.optimize()

    check("result.frame_poses is dict", isinstance(result.frame_poses, dict))
    check("result.final_error is float", isinstance(result.final_error, float))
    check("result.iterations is int", isinstance(result.iterations, int))
    check("result.pose() returns (4,4)",
          all(result.pose(k).shape == (4, 4) for k in result.frame_poses))
    check("duplicate add_frame is ignored", graph.n_nodes == 3)
    graph.add_frame(0, np.eye(4))  # silently skipped — node already exists
    check("n_nodes unchanged after duplicate add", graph.n_nodes == 3)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    np.random.seed(42)

    test_single_frame(args.verbose)
    test_chain_recovery(args.verbose)
    test_loop_closure(args.verbose)
    test_warm_start(args.verbose)
    test_result_interface()

    header("All checks passed")


if __name__ == "__main__":
    main()
