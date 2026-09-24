import torch

from trex.panel import (
    NuclearNormMatrixCompletion,
    SyntheticDID,
    collapsed_form,
    matrix_completion_lambda_max,
    simplex_least_squares_fw,
)


def _make_low_rank_panel(seed=0, n=36, t=28, rank=2, noise=0.03, dtype=torch.float64):
    torch.manual_seed(seed)
    U = torch.randn(n, rank, dtype=dtype)
    V = torch.randn(t, rank, dtype=dtype)
    unit_fe = torch.linspace(-1.0, 1.0, n, dtype=dtype)
    time_fe = 0.5 * torch.sin(torch.linspace(-2.0, 2.0, t, dtype=dtype))
    mean = U @ V.T + unit_fe[:, None] + time_fe[None, :]
    y = mean + noise * torch.randn(n, t, dtype=dtype)
    return y, mean


def test_nuclear_norm_matrix_completion_improves_holdout_prediction():
    y, mean = _make_low_rank_panel(seed=1)
    torch.manual_seed(2)
    mask = torch.rand_like(y) < 0.65
    assert torch.any(~mask)

    baseline = torch.zeros_like(y)
    row_mean = torch.where(mask, y, torch.zeros_like(y)).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    col_mean = torch.where(mask, y, torch.zeros_like(y)).sum(dim=0) / mask.sum(dim=0).clamp_min(1)
    grand = y[mask].mean()
    baseline = row_mean[:, None] + col_mean[None, :] - grand

    model = NuclearNormMatrixCompletion(lambda_fraction=0.08, maxiter=300, tol=1e-7, device="cpu")
    model.fit(y, mask=mask)
    pred = model.predict()

    holdout = ~mask
    mc_mse = torch.mean((pred[holdout] - mean[holdout]) ** 2)
    baseline_mse = torch.mean((baseline[holdout] - mean[holdout]) ** 2)

    assert model.result_.iterations <= model.maxiter
    assert model.result_.singular_values.gt(1e-8).sum() <= 8
    assert mc_mse < 0.55 * baseline_mse


def test_matrix_completion_lambda_max_zeroes_low_rank_component():
    y, _ = _make_low_rank_panel(seed=3, n=18, t=14)
    mask = torch.ones_like(y, dtype=torch.bool)
    lam_max = matrix_completion_lambda_max(y, mask)

    model = NuclearNormMatrixCompletion(lambda_L=float(lam_max * 1.05), maxiter=80, tol=1e-8)
    model.fit(y, mask=mask)

    assert torch.linalg.norm(model.result_.low_rank) < 1e-6


def test_simplex_least_squares_fw_recovers_sparse_convex_weights():
    torch.manual_seed(4)
    A = torch.randn(40, 6, dtype=torch.float64)
    x_true = torch.tensor([0.0, 0.35, 0.0, 0.5, 0.15, 0.0], dtype=torch.float64)
    b = A @ x_true

    x_hat, vals = simplex_least_squares_fw(A, b, zeta=1e-8, intercept=False, maxiter=6000)

    assert torch.all(x_hat >= -1e-10)
    assert torch.allclose(x_hat.sum(), torch.tensor(1.0, dtype=x_hat.dtype), atol=1e-8)
    # Frank-Wolfe is intentionally reference-like and sparse; on a dense random
    # design it converges sublinearly but should still reach a small residual.
    assert torch.norm(A @ x_hat - b) < 5e-2
    assert vals[-1] <= vals[0]


def test_collapsed_form_matches_manual_synthdid_shape_and_entries():
    Y = torch.arange(5 * 6, dtype=torch.float64).reshape(5, 6)
    Yc = collapsed_form(Y, N0=3, T0=4)

    assert Yc.shape == (4, 5)
    assert torch.allclose(Yc[:3, :4], Y[:3, :4])
    assert torch.allclose(Yc[:3, 4], Y[:3, 4:].mean(dim=1))
    assert torch.allclose(Yc[3, :4], Y[3:, :4].mean(dim=0))
    assert torch.allclose(Yc[3, 4], Y[3:, 4:].mean())


def test_synthetic_did_estimate_close_to_known_constant_effect():
    torch.manual_seed(5)
    n0, n1, t0, t1 = 45, 8, 35, 8
    n, t = n0 + n1, t0 + t1
    rank = 2
    U = torch.randn(n, rank, dtype=torch.float64)
    V = torch.randn(t, rank, dtype=torch.float64)
    unit_fe = torch.randn(n, dtype=torch.float64)
    time_fe = torch.linspace(-1.0, 1.0, t, dtype=torch.float64)
    tau = 1.25
    Y0 = U @ V.T + unit_fe[:, None] + time_fe[None, :] + 0.05 * torch.randn(n, t, dtype=torch.float64)
    Y = Y0.clone()
    Y[n0:, t0:] += tau

    est = SyntheticDID(maxiter=2500, min_decrease=1e-9, sparsify=True).fit(Y, n0, t0).result_

    assert torch.all(est.omega >= -1e-10)
    assert torch.all(est.lambda_ >= -1e-10)
    assert torch.allclose(est.omega.sum(), torch.tensor(1.0, dtype=Y.dtype), atol=1e-8)
    assert torch.allclose(est.lambda_.sum(), torch.tensor(1.0, dtype=Y.dtype), atol=1e-8)
    assert abs(float(est.estimate - tau)) < 0.35


def _load_california_prop99_tensor():
    import csv
    from pathlib import Path

    path = Path(__file__).parent / "data" / "california_prop99.csv"
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rows.append((row["State"], int(row["Year"]), float(row["PacksPerCapita"]), int(row["treated"])))
    states = sorted({r[0] for r in rows})
    treated_states = sorted({r[0] for r in rows if r[3] == 1})
    states = [s for s in states if s not in treated_states] + treated_states
    years = sorted({r[1] for r in rows})
    by_key = {(s, y): (packs, treated) for s, y, packs, treated in rows}
    Y = torch.empty((len(states), len(years)), dtype=torch.float64)
    W = torch.empty_like(Y, dtype=torch.bool)
    for i, state in enumerate(states):
        for j, year in enumerate(years):
            Y[i, j], W[i, j] = by_key[(state, year)]
    return Y, int((~W.any(dim=1)).sum()), int((~W.any(dim=0)).sum())


def test_panel_estimates_reproduce_synthdid_california_point_estimates():
    from trex.panel import panel_estimates

    Y, N0, T0 = _load_california_prop99_tensor()
    estimates = panel_estimates(
        Y,
        N0,
        T0,
        methods=[
            "DID",
            "Synthetic Control (SC)",
            "Synthetic DID (SDID)",
            "Time Weighted DID",
            "SDID (No Intercept)",
            "SC with FEs (DIFP)",
            "SC (Regularized)",
            "DIFP (Regularized)",
        ],
        sdid_kwargs={"maxiter": 10_000, "min_decrease": 1e-5, "sparsify": True},
    )
    expected = {
        "DID": -27.34911,
        "Synthetic Control (SC)": -19.61966,
        "Synthetic DID (SDID)": -15.60383,
        "Time Weighted DID": -19.77199,
        "SDID (No Intercept)": -18.75256,
        "SC with FEs (DIFP)": -11.10464,
        "SC (Regularized)": -21.71706,
        "DIFP (Regularized)": -16.12120,
    }
    for name, target in expected.items():
        assert abs(float(estimates[name]) - target) < 2e-3
