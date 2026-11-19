"""pytest wrapper: `pytest -q` from the repository root.

Fast tests only (no MESA re-reduction beyond loading the histories once)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ztf_j1239_simulations as sim  # noqa: E402


@pytest.fixture(scope="module")
def ds():
    return sim.load_or_build_age_grid_dataset(strict=True)   # fails if ./runs is missing


def test_builtin_verification_suite():
    sim.run_verification_tests()


def test_strict_dataset_provenance(ds):
    p = ds["provenance"]
    assert p["source"].startswith("raw_mesa_histories")
    assert p["n_run_dirs"] == 266


def test_cond03_agreement_1gyr(ds):
    r_mesa = np.array([ds["snap_table"]["1"][m]["R"] for m in sim.MASSES_MJUP])
    r_cond = sim.cond03_radius_rj(ROOT / "data" / "COND03_1Gyr.csv", sim.MASSES_MJUP)
    assert np.max(np.abs(r_mesa / r_cond - 1)) < 0.02


def test_roche_exponent_matches_numerical_derivative():
    # zeta_L = Z0 + f S must equal d ln R_L / d ln Md for isotropic point-source loss; check Z0 numerically
    q, h = 0.7, 1e-5
    f = lambda x: float(sim.eggleton_roche_lobe(x))
    g = (np.log(f(q * (1 + h))) - np.log(f(q * (1 - h)))) / (np.log(1 + h) - np.log(1 - h))
    assert abs(g - float(sim.dlnf_dlnq(q))) < 1e-6


def test_results_match_regenerated_values(ds):
    res = json.load(open(ROOT / "results" / "ztf_j1239_simulation_results.json"))
    zfn, rfn, _ = sim.make_age_interpolators(ds)
    row = [r for r in res["age_scaling"] if r["age_gyr"] == 3.0][0]
    assert abs(rfn(80.0, 3.0) - row["Ra_80_RJ"]) < 1e-9
    qc = sim.critical_mass_ratio(lambda md: zfn(md, 3.0), Ma_mjup=80.0)
    assert abs(qc - row["q_crit_f0_80"]) < 1e-9


def test_binary_escape_speed_exceeds_two_body():
    assert sim.binary_escape_velocity(60, 80, 0.9) > sim.escape_velocity(80, 0.9)


def test_stream_impact_speed_age_trend(ds):
    _, rfn, _ = sim.make_age_interpolators(ds)
    v = [sim.stream_energetics(60, 80, rfn(80.0, a))["v_imp"] for a in (1.0, 2.0, 5.0)]
    assert v[0] < v[1] < v[2]
