"""
Test da3_slam.backend.processing.factor_graph (Sim3 pose graph).

All tests use synthetic submaps — no images, models, or GPU required.

Tests:
  1. Sim3 Lie algebra  — exp/log roundtrip, inverse, zero-residual condition
  2. Single submap     — prior pins pose at identity, scale=1
  3. Chain of submaps  — positions recover cumulative translations
  4. Scale constraint  — loop closure with scale != 1 moves optimised scale
  5. Warm-start        — second optimize() converges in fewer Jacobian evals
  6. Result interface  — poses, scales, final_error, iterations are correct types

Usage:
    python scripts/test_pose_graph.py
    python scripts/test_pose_graph.py --verbose   # show scipy LM output
"""

import argparse
import time

import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true",
                        help="Show scipy LM optimisation output")
    return parser.parse_args()


def header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


def check(label: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        raise AssertionError(f"FAIL: {label}")


# ── synthetic data factories ──────────────────────────────────────────────────

def make_submap(idx: int, n_points: int = 100):
    """Minimal synthetic Submap — no real depth or images."""
    from da3_slam.backend.inference.submap import Submap, Frame

    pts = np.random.randn(n_points, 3).astype(np.float32)
    pts[:, 2] = np.abs(pts[:, 2]) + 1.0   # z > 0

    frame = Frame(
        seq_idx=idx * 10,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        points_cam=pts,
        points_world=pts.copy(),
        colors=np.zeros((n_points, 3), dtype=np.uint8),
        extrinsic=np.eye(4, dtype=np.float32),
        intrinsic=np.eye(3, dtype=np.float32),
    )
    sm = Submap(idx=idx)
    sm.frames.append(frame)
    return sm


def se3_alignment(t: np.ndarray, R: np.ndarray | None = None):
    """Anchor-frame SE3 alignment (scale = 1)."""
    from da3_slam.backend.processing.alignment import AlignmentResult
    T = np.eye(4, dtype=np.float32)
    if R is not None:
        T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return AlignmentResult(world_b_to_world_a=T, method="anchor", scale=1.0)


def sim3_alignment(t: np.ndarray, s: float, R: np.ndarray | None = None):
    """Sim3 ICP alignment: T[:3,:3] = s·R, scale = s."""
    from da3_slam.backend.processing.alignment import AlignmentResult
    T = np.eye(4, dtype=np.float32)
    R_mat = R if R is not None else np.eye(3, dtype=np.float32)
    T[:3, :3] = (s * R_mat).astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return AlignmentResult(world_b_to_world_a=T, method="icp", scale=s)


def default_noise():
    from da3_slam.backend.processing.factor_graph import NoiseConfig
    return NoiseConfig(
        prior_rotation_sigma=1e-6,
        prior_translation_sigma=1e-6,
        between_rotation_sigma=1e-3,
        between_translation_sigma=1e-2,
        between_scale_sigma=0.10,
        loop_rotation_sigma=5e-3,
        loop_translation_sigma=5e-2,
        loop_scale_sigma=0.05,
    )


# ── tests ─────────────────────────────────────────────────────────────────────

def test_sim3_algebra():
    from da3_slam.backend.processing.factor_graph import _sim3_exp, _sim3_log, _sim3_inv

    header("1 — Sim3 Lie algebra")

    # exp → log roundtrip
    for i in range(6):
        xi = np.random.randn(7) * 0.4
        xi[6] = np.random.uniform(-0.8, 0.8)
        xi2 = _sim3_log(_sim3_exp(xi))
        err = float(np.linalg.norm(xi - xi2))
        check(f"exp→log roundtrip [{i}]  err={err:.1e}", err < 1e-12)

    # T @ inv(T) = I
    for i in range(3):
        xi = np.random.randn(7) * 0.4
        T = _sim3_exp(xi)
        err = float(np.linalg.norm(T @ _sim3_inv(T) - np.eye(4)))
        check(f"T @ inv(T) = I [{i}]  err={err:.1e}", err < 1e-12)

    # Pure scale: exp([0,0,0, 0,0,0, λ]) should give s=exp(λ), R=I, t=0
    for lam in [0.3, -0.5, 1.0]:
        xi = np.zeros(7)
        xi[6] = lam
        T = _sim3_exp(xi)
        s = float(np.cbrt(np.linalg.det(T[:3, :3])))
        err_s = abs(s - np.exp(lam))
        err_R = float(np.linalg.norm(T[:3, :3] / s - np.eye(3)))
        check(f"pure scale λ={lam:+.1f}: s={s:.4f}≈{np.exp(lam):.4f}", err_s < 1e-12)
        check(f"pure scale λ={lam:+.1f}: R=I", err_R < 1e-12)

    # Between-factor zero-residual: when T_meas = T_pred, error = 0
    xi_a = np.random.randn(7) * 0.3
    xi_b = np.random.randn(7) * 0.3
    T_a, T_b = _sim3_exp(xi_a), _sim3_exp(xi_b)
    T_meas = _sim3_inv(T_a) @ T_b
    T_pred = _sim3_inv(T_a) @ T_b
    e = float(np.linalg.norm(_sim3_log(_sim3_inv(T_meas) @ T_pred)))
    check(f"between-factor zero residual  err={e:.1e}", e < 1e-12)


def test_single_submap(verbose: bool):
    from da3_slam.backend.processing.factor_graph import PoseGraph

    header("2 — Single submap: prior pins identity")

    graph = PoseGraph(default_noise())
    graph.add_submap(make_submap(0))

    check("n_nodes = 1", graph.n_nodes == 1)
    check("n_factors = 1 (prior only)", graph.n_factors == 1)

    result = graph.optimize(verbose=verbose)

    check("result has 1 pose", len(result.poses) == 1)
    check("result has 1 scale", len(result.scales) == 1)
    check("final_error is finite", np.isfinite(result.final_error))

    s0 = result.scale(0)
    T0 = result.pose(0)
    R0 = T0[:3, :3] / s0

    check(f"scale(0) ≈ 1.0  got={s0:.6f}", abs(s0 - 1.0) < 1e-3)
    check("rotation ≈ identity", np.allclose(R0, np.eye(3), atol=1e-3))
    check("translation ≈ zero", np.allclose(T0[:3, 3], 0, atol=1e-3))
    print(f"  scale={s0:.6f}  cost={result.final_error:.2e}  iters={result.iterations}")


def test_chain_recovery(verbose: bool):
    from da3_slam.backend.processing.factor_graph import PoseGraph

    header("3 — Chain of 3 submaps: position recovery")

    t_01 = np.array([0.50,  0.00, 0.0])
    t_12 = np.array([0.50,  0.10, 0.0])
    t_23 = np.array([0.30, -0.05, 0.0])

    submaps = [make_submap(i) for i in range(4)]
    translations = [t_01, t_12, t_23]

    graph = PoseGraph(default_noise())
    graph.add_submap(submaps[0])
    for i in range(3):
        graph.add_submap(submaps[i + 1], se3_alignment(translations[i]))

    check("n_nodes = 4", graph.n_nodes == 4)
    check("n_factors = 4 (1 prior + 3 between)", graph.n_factors == 4)

    result = graph.optimize(verbose=verbose)

    cumulative = np.zeros(3)
    for i, t in enumerate(translations):
        cumulative += t
        pos_opt = result.pose(i + 1)[:3, 3]
        err = float(np.linalg.norm(pos_opt - cumulative))
        check(f"position({i+1}) matches cumulative  err={err:.4f}m", err < 0.01)
        print(f"  node {i+1}: pos={pos_opt.round(4)}  expected={cumulative.round(4)}")

    # Without loop closure, scales stay near 1
    for i in range(4):
        s = result.scale(i)
        check(f"scale({i}) ≈ 1.0 (no loop)  got={s:.4f}", abs(s - 1.0) < 0.05)


def test_scale_constraint(verbose: bool):
    from da3_slam.backend.processing.factor_graph import NoiseConfig, PoseGraph

    header("4 — Loop closure scale constraint")

    # Very loose between-scale, tight loop-scale: loop closure dominates
    noise = NoiseConfig(
        prior_rotation_sigma=1e-6,
        prior_translation_sigma=1e-6,
        between_rotation_sigma=1e-3,
        between_translation_sigma=1e-2,
        between_scale_sigma=1.00,   # very loose — scales can drift freely
        loop_rotation_sigma=5e-3,
        loop_translation_sigma=5e-2,
        loop_scale_sigma=0.01,      # tight — loop closure owns the scale
    )

    t_01 = np.array([0.5, 0.0, 0.0])
    t_12 = np.array([0.5, 0.1, 0.0])

    graph = PoseGraph(noise)
    sm0, sm1, sm2 = make_submap(0), make_submap(1), make_submap(2)
    graph.add_submap(sm0)
    graph.add_submap(sm1, se3_alignment(t_01))
    graph.add_submap(sm2, se3_alignment(t_12))

    # Loop closure: submap 2 is seen as 80% the metric scale of submap 0
    target_scale = 0.8
    t_02 = t_01 + t_12
    graph.add_loop_closure(0, 2, sim3_alignment(t_02, s=target_scale))

    check("n_factors = 4 (1 prior + 2 between + 1 loop)", graph.n_factors == 4)

    result = graph.optimize(verbose=verbose)

    s2 = result.scale(2)
    s0 = result.scale(0)

    check(f"scale(0) ≈ 1.0 (pinned by prior)  got={s0:.4f}", abs(s0 - 1.0) < 1e-3)
    check(f"scale(2) converged toward {target_scale}  got={s2:.4f}",
          abs(s2 - target_scale) < 0.05)
    print(f"  target={target_scale}  scale(0)={s0:.4f}  scale(2)={s2:.4f}")

    # Verify scale(2) actually moved from 1.0 toward target
    check("scale(2) < 0.90 (moved from 1.0)", s2 < 0.90)


def test_warm_start(verbose: bool):
    from da3_slam.backend.processing.factor_graph import PoseGraph

    header("5 — Warm-start")

    translations = [np.array([0.4 * (i + 1), 0.0, 0.0]) for i in range(6)]
    submaps = [make_submap(i) for i in range(7)]

    graph = PoseGraph(default_noise())
    graph.add_submap(submaps[0])
    for i in range(6):
        graph.add_submap(submaps[i + 1], se3_alignment(translations[i]))

    t0 = time.time()
    r1 = graph.optimize(verbose=verbose)
    t1 = time.time() - t0

    t0 = time.time()
    r2 = graph.optimize(verbose=verbose)    # warm-start from r1
    t2 = time.time() - t0

    check("warm-start result is finite", np.isfinite(r2.final_error))
    check("warm-start error ≤ cold-start error",
          r2.final_error <= r1.final_error + 1e-6)
    check("warm-start uses ≤ Jacobian evals as cold-start",
          r2.iterations <= r1.iterations)

    print(f"  cold: {r1.iterations} Jac evals  {t1:.3f}s  cost={r1.final_error:.2e}")
    print(f"  warm: {r2.iterations} Jac evals  {t2:.3f}s  cost={r2.final_error:.2e}")


def test_result_interface():
    from da3_slam.backend.processing.factor_graph import PoseGraph

    header("6 — OptimizationResult interface")

    graph = PoseGraph(default_noise())
    for i in range(3):
        if i == 0:
            graph.add_submap(make_submap(i))
        else:
            graph.add_submap(make_submap(i), se3_alignment(np.array([float(i), 0., 0.])))
    result = graph.optimize()

    check("result.poses is dict", isinstance(result.poses, dict))
    check("result.scales is dict", isinstance(result.scales, dict))
    check("result.final_error is float", isinstance(result.final_error, float))
    check("result.iterations is int", isinstance(result.iterations, int))
    check("result.pose() returns (4,4)",
          all(result.pose(k).shape == (4, 4) for k in result.poses))
    check("result.scale() returns float",
          all(isinstance(result.scale(k), float) for k in result.poses))
    check("all scales > 0",
          all(s > 0 for s in result.scales.values()))
    check("all rotation blocks are SO3 (det > 0)",
          all(np.linalg.det(result.poses[k][:3, :3]) > 0 for k in result.poses))

    for k in sorted(result.poses):
        s = result.scale(k)
        T = result.pose(k)
        R = T[:3, :3] / s
        det_err = abs(np.linalg.det(R) - 1.0)
        check(f"submap {k}: det(R) ≈ 1.0  err={det_err:.2e}", det_err < 1e-3)
        print(f"  submap {k}: scale={s:.4f}  pos={T[:3,3].round(4)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    np.random.seed(42)

    test_sim3_algebra()
    test_single_submap(args.verbose)
    test_chain_recovery(args.verbose)
    test_scale_constraint(args.verbose)
    test_warm_start(args.verbose)
    test_result_interface()

    header("All checks passed")


if __name__ == "__main__":
    main()
