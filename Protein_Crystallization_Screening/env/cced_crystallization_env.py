from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

import gymnasium as gym
from gymnasium import spaces


# -----------------------------
# 1) 规范定义：协议、约束、奖励景观
# -----------------------------
@dataclass
class ProtocolSpec:
    """Protocol-specific convex envelope: G u <= h, where u=[p1..pR, T, tau]."""
    name: str
    G: np.ndarray  # (m, d)
    h: np.ndarray  # (m,)
    peaks_mu: np.ndarray      # (M, d) favorable regions (multimodal)
    peaks_sigma: np.ndarray   # (M, d) diagonal std
    peaks_w: np.ndarray       # (M,) weights sum to 1
    disc_cost: float = 0.0    # optional discrete cost (e.g., seeding more expensive)


@dataclass
class CrystalCCEDSpec:
    seed: int = 0
    K: int = 5                # protocols
    R: int = 4                # fractions: PEG, salt, buffer, additive
    batch_size: int = 2       # droplets per step
    horizon: int = 25

    p_max: float = 0.85
    T_bounds: Tuple[float, float] = (0.0, 1.0)    # normalized (map to [4C, 25C] if you want)
    tau_bounds: Tuple[float, float] = (0.0, 1.0)  # normalized (map to [6h, 72h] if you want)

    yield_noise_std: float = 0.01
    cost_lambda: float = 0.05
    cost_add: float = 0.20
    cost_T: float = 0.05
    cost_tau: float = 0.03

    protocols: List[ProtocolSpec] = None


def make_protein_crystallization_spec(seed: int = 0, batch_size: int = 2, horizon: int = 25) -> CrystalCCEDSpec:
    """
    A concrete non-industrial scenario: high-throughput protein crystallization screening.
    Hybrid action: protocol (discrete) + droplet composition & incubation (continuous).
    """
    rng = np.random.default_rng(seed)

    K = 5
    R = 4
    d = R + 2
    T_idx, tau_idx = R, R + 1

    def rows(*ineqs):
        G = np.vstack([v for v, _h in ineqs]).astype(float)
        h = np.array([_h for _v, _h in ineqs], dtype=float)
        return G, h

    # ---- Protocol 0: PEG vapor diffusion (PEG-rich, limited salt/additive)
    ineqs = []
    ineqs.append((np.array([-1, 0, 0, 0, 0, 0]), -0.35))      # p_PEG >= 0.35  -> -p_PEG <= -0.35
    ineqs.append((np.array([0, 1, 0, 0, 0, 0]), 0.25))        # p_salt <= 0.25
    ineqs.append((np.array([0, 0, 0, 1, 0, 0]), 0.12))        # p_add <= 0.12
    v = np.zeros(d); v[T_idx] = 1.0; v[tau_idx] = 0.5
    ineqs.append((v, 1.30))                                   # T + 0.5*tau <= 1.30
    G0, h0 = rows(*ineqs)

    peaks0 = np.array([
        [0.55, 0.10, 0.27, 0.08, 0.25, 0.75],
        [0.45, 0.18, 0.30, 0.07, 0.20, 0.65],
        [0.60, 0.12, 0.23, 0.05, 0.15, 0.80],
    ])

    # ---- Protocol 1: Salt screen (salt-rich, limited PEG/additive, moderate temperature)
    ineqs = []
    ineqs.append((np.array([0, -1, 0, 0, 0, 0]), -0.35))      # p_salt >= 0.35
    ineqs.append((np.array([1, 0, 0, 0, 0, 0]), 0.25))        # p_PEG <= 0.25
    ineqs.append((np.array([0, 0, 0, 1, 0, 0]), 0.12))        # p_add <= 0.12
    v = np.zeros(d); v[T_idx] = 1.0
    ineqs.append((v, 0.70))                                   # T <= 0.70
    G1, h1 = rows(*ineqs)

    peaks1 = np.array([
        [0.15, 0.55, 0.22, 0.08, 0.30, 0.55],
        [0.10, 0.45, 0.35, 0.10, 0.25, 0.60],
        [0.20, 0.50, 0.20, 0.10, 0.40, 0.50],
    ])

    # ---- Protocol 2: Organic precipitant (buffer-rich, low additive, warmer)
    ineqs = []
    ineqs.append((np.array([0, 0, 0, 1, 0, 0]), 0.08))        # p_add <= 0.08
    ineqs.append((np.array([0, 0, -1, 0, 0, 0]), -0.25))      # p_buffer >= 0.25
    v = np.zeros(d); v[T_idx] = -1.0
    ineqs.append((v, -0.35))                                  # T >= 0.35
    v = np.zeros(d); v[tau_idx] = 1.0
    ineqs.append((v, 0.80))                                   # tau <= 0.80
    G2, h2 = rows(*ineqs)

    peaks2 = np.array([
        [0.25, 0.15, 0.55, 0.05, 0.55, 0.45],
        [0.20, 0.10, 0.62, 0.08, 0.45, 0.55],
        [0.18, 0.20, 0.57, 0.05, 0.60, 0.35],
    ])

    # ---- Protocol 3: Microseeding (time-limited, buffer-rich, PEG in mid-range)
    ineqs = []
    v = np.zeros(d); v[tau_idx] = 1.0
    ineqs.append((v, 0.55))                                   # tau <= 0.55
    ineqs.append((np.array([0, 0, -1, 0, 0, 0]), -0.25))      # p_buffer >= 0.25
    ineqs.append((np.array([-1, 0, 0, 0, 0, 0]), -0.20))      # p_PEG >= 0.20
    ineqs.append((np.array([ 1, 0, 0, 0, 0, 0]),  0.60))      # p_PEG <= 0.60
    ineqs.append((np.array([0, 0, 0, 1, 0, 0]), 0.15))        # p_add <= 0.15
    G3, h3 = rows(*ineqs)

    peaks3 = np.array([
        [0.35, 0.15, 0.40, 0.10, 0.30, 0.40],
        [0.25, 0.20, 0.45, 0.10, 0.25, 0.50],
        [0.40, 0.10, 0.40, 0.10, 0.20, 0.35],
    ])

    # ---- Protocol 4: Additive-rich screen (higher additive, limited salt, moderate time)
    ineqs = []
    ineqs.append((np.array([0, 0, 0, -1, 0, 0]), -0.15))      # p_add >= 0.15
    ineqs.append((np.array([0, 0, -1, 0, 0, 0]), -0.15))      # p_buffer >= 0.15
    ineqs.append((np.array([0, 1, 0, 0, 0, 0]), 0.25))        # p_salt <= 0.25
    v = np.zeros(d); v[tau_idx] = 1.0
    ineqs.append((v, 0.75))                                   # tau <= 0.75
    G4, h4 = rows(*ineqs)

    peaks4 = np.array([
        [0.30, 0.15, 0.35, 0.20, 0.35, 0.55],
        [0.25, 0.20, 0.30, 0.25, 0.30, 0.60],
        [0.28, 0.10, 0.40, 0.22, 0.40, 0.50],
    ])

    def mk_proto(name: str, G: np.ndarray, h: np.ndarray, peaks: np.ndarray, disc_cost: float):
        M, d2 = peaks.shape
        assert d2 == d
        sigma = np.full((M, d), 0.09, dtype=float)
        w = np.ones(M, dtype=float) / M
        return ProtocolSpec(name=name, G=G, h=h, peaks_mu=peaks, peaks_sigma=sigma, peaks_w=w, disc_cost=disc_cost)

    protocols = [
        mk_proto("PEG_vapor", G0, h0, peaks0, disc_cost=0.02),
        mk_proto("Salt_screen", G1, h1, peaks1, disc_cost=0.01),
        mk_proto("Organic_precip", G2, h2, peaks2, disc_cost=0.015),
        mk_proto("Microseeding", G3, h3, peaks3, disc_cost=0.03),
        mk_proto("Additive_rich", G4, h4, peaks4, disc_cost=0.025),
    ]

    return CrystalCCEDSpec(
        seed=seed,
        K=K,
        R=R,
        batch_size=batch_size,
        horizon=horizon,
        protocols=protocols,
    )


# -----------------------------
# 2) Logic-to-manifold projector: LAP + convex QP projection (SLSQP)
# -----------------------------
class HybridLogicProjector:
    """
    Latent action z -> (discrete assignment via Hungarian) + per-slot convex projection (SLSQP).

    Latent format:
      For each droplet slot j:
        [K logits] + [d continuous intent]
    Total dim = B*(K+d)
    """
    def __init__(self, spec: CrystalCCEDSpec):
        self.spec = spec
        self.K = spec.K
        self.R = spec.R
        self.d = spec.R + 2
        self.B = spec.batch_size
        self.T_idx = self.R
        self.tau_idx = self.R + 1

    def _parse_latent(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = np.asarray(z, dtype=float).reshape(-1)
        expected = self.B * (self.K + self.d)
        if z.size != expected:
            raise ValueError(f"latent dim {z.size} != expected {expected} = B*(K+d)")
        tmp = z.reshape(self.B, self.K + self.d)
        logits = tmp[:, :self.K]
        cont = tmp[:, self.K:]
        return logits, cont

    def _assign_protocols(self, logits: np.ndarray, enabled: Optional[np.ndarray] = None) -> np.ndarray:
        B, K = logits.shape
        if enabled is not None:
            enabled = np.asarray(enabled, dtype=bool)

        # If B <= K, use Hungarian for unique assignment (resource-limited parallel screening)
        if B <= K:
            cost = -logits.copy()
            if enabled is not None:
                for k in range(K):
                    if not enabled[k]:
                        cost[:, k] = 1e6
            row_ind, col_ind = linear_sum_assignment(cost)
            k_vec = np.zeros(B, dtype=int)
            for r, c in zip(row_ind, col_ind):
                k_vec[r] = int(c)
            return k_vec

        # If B > K, allow duplicates (greedy)
        k_vec = []
        for j in range(B):
            order = np.argsort(-logits[j])
            for k in order:
                if enabled is None or enabled[int(k)]:
                    k_vec.append(int(k))
                    break
        return np.asarray(k_vec, dtype=int)

    def _feasible_default(self) -> np.ndarray:
        p = np.ones(self.R) / self.R
        T = float(np.mean(self.spec.T_bounds))
        tau = float(np.mean(self.spec.tau_bounds))
        return np.concatenate([p, [T, tau]])

    def _project_continuous(self, k: int, z_u: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve min ||u - z_u||^2 s.t. simplex + bounds + G_k u <= h_k via SLSQP."""
        sp = self.spec
        proto = sp.protocols[int(k)]
        d = self.d
        R = self.R

        def f(u):
            diff = u - z_u
            return 0.5 * float(diff @ diff)

        def grad(u):
            return (u - z_u)

        # Equality: sum(p)=1
        cons = [{
            "type": "eq",
            "fun": lambda u: float(np.sum(u[:R]) - 1.0),
            "jac": lambda u: np.concatenate([np.ones(R), np.zeros(d - R)])
        }]

        # Inequalities: h - G u >= 0
        if proto.G is not None and proto.G.size > 0:
            for row, hi in zip(proto.G, proto.h):
                row = row.astype(float)
                hi = float(hi)
                cons.append({
                    "type": "ineq",
                    "fun": lambda u, row=row, hi=hi: float(hi - row @ u),
                    "jac": lambda u, row=row, hi=hi: -row
                })

        # Bounds cover p>=0, p<=p_max, T,tau ranges
        bounds = [(0.0, sp.p_max)] * R + [sp.T_bounds, sp.tau_bounds]

        # Initial guess: clipped + normalized simplex
        x0 = np.clip(z_u, [b[0] for b in bounds], [b[1] for b in bounds])
        p = x0[:R].clip(0.0, sp.p_max)
        p = p / (np.sum(p) + 1e-12)
        x0[:R] = p

        res = minimize(
            f, x0,
            method="SLSQP",
            jac=grad,
            constraints=cons,
            bounds=bounds,
            options={"ftol": 1e-9, "maxiter": 200, "disp": False},
        )

        if (not res.success) or (res.x is None):
            return self._feasible_default(), {"status": "fail", "message": str(res.message)}
        return res.x.astype(float), {"status": "ok", "nit": int(res.nit)}

    def project(self, z: np.ndarray, enabled_protocols: Optional[np.ndarray] = None):
        logits, cont = self._parse_latent(z)
        k_vec = self._assign_protocols(logits, enabled=enabled_protocols)

        u_mat = np.zeros((self.B, self.d), dtype=float)
        infos = []
        for j in range(self.B):
            u, info = self._project_continuous(int(k_vec[j]), cont[j])
            u_mat[j] = u
            infos.append({"slot": j, "k": int(k_vec[j]), **info})
        return k_vec, u_mat, {"slots": infos}


# -----------------------------
# 3) Base env: explicit hybrid action -> reward (quality score)
# -----------------------------
class ProteinCrystallizationBaseEnv(gym.Env):
    """
    Explicit hybrid-action environment:
      action = {"k": (B,), "u": (B,d)}
    Observation: [t_norm, best_quality, last_mean_quality]
    Reward: mean_quality - lambda * (continuous+discrete cost)
    """
    metadata = {"render_modes": []}

    def __init__(self, spec: CrystalCCEDSpec):
        super().__init__()
        self.spec = spec
        self.K = spec.K
        self.R = spec.R
        self.d = spec.R + 2
        self.B = spec.batch_size
        self.horizon = spec.horizon

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Dict({
            "k": spaces.MultiDiscrete([self.K] * self.B),
            "u": spaces.Box(low=-np.inf, high=np.inf, shape=(self.B, self.d), dtype=np.float32),
        })

        self._rng = np.random.default_rng(spec.seed)
        self._t = 0
        self._best = 0.0
        self._last_mean = 0.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._best = 0.0
        self._last_mean = 0.0
        obs = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return obs, {}

    def _quality(self, k: int, u: np.ndarray) -> float:
        proto = self.spec.protocols[int(k)]
        vals = []
        for mu, sig, w in zip(proto.peaks_mu, proto.peaks_sigma, proto.peaks_w):
            z = (u - mu) / (sig + 1e-12)
            vals.append(float(w * np.exp(-0.5 * np.sum(z * z))))
        q = float(np.sum(vals))
        q += float(self._rng.normal(0.0, self.spec.yield_noise_std))
        return float(np.clip(q, 0.0, 1.0))

    def _cost(self, k: int, u: np.ndarray) -> float:
        p_add = float(u[3])      # additive fraction
        T = float(u[self.R])
        tau = float(u[self.R + 1])
        return float(self.spec.protocols[int(k)].disc_cost + self.spec.cost_add * p_add + self.spec.cost_T * T + self.spec.cost_tau * tau)

    def step(self, action: Dict[str, Any]):
        k_vec = np.asarray(action["k"], dtype=int).reshape(self.B)
        u_mat = np.asarray(action["u"], dtype=float).reshape(self.B, self.d)

        qualities = []
        costs = []
        for j in range(self.B):
            q = self._quality(int(k_vec[j]), u_mat[j])
            c = self._cost(int(k_vec[j]), u_mat[j])
            qualities.append(q)
            costs.append(c)

        mean_q = float(np.mean(qualities))
        mean_c = float(np.mean(costs))
        reward = mean_q - self.spec.cost_lambda * mean_c

        self._best = max(self._best, mean_q)
        self._last_mean = mean_q
        self._t += 1

        terminated = (self._t >= self.horizon)
        truncated = False
        obs = np.array([self._t / self.horizon, self._best, self._last_mean], dtype=np.float32)
        info = {"mean_quality": mean_q, "mean_cost": mean_c, "best_quality": self._best, "t": self._t}
        return obs, reward, terminated, truncated, info


# -----------------------------
# 4) Projected env: latent z -> projector -> explicit action (guaranteed feasible)
# -----------------------------
class ProjectedProteinCrystallizationEnv(gym.Env):
    """
    Action is latent continuous vector z with dim = B*(K+d).
    The projector maps z -> feasible (k,u). We track CVR to verify 0 violation.
    """
    metadata = {"render_modes": []}

    def __init__(self, base_env: ProteinCrystallizationBaseEnv, projector: HybridLogicProjector,
                 enabled_protocols: Optional[np.ndarray] = None):
        super().__init__()
        self.base_env = base_env
        self.projector = projector
        self.spec = base_env.spec
        self.enabled_protocols = enabled_protocols

        self.observation_space = base_env.observation_space
        dim = self.spec.batch_size * (self.spec.K + (self.spec.R + 2))
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

        self._violations = 0
        self._steps = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._violations = 0
        self._steps = 0
        return self.base_env.reset(seed=seed, options=options)

    def _check_feasible(self, k: int, u: np.ndarray) -> bool:
        sp = self.spec
        R = sp.R
        p = u[:R]
        T = float(u[R])
        tau = float(u[R + 1])

        if np.any(p < -1e-6) or np.any(p > sp.p_max + 1e-6):
            return False
        if abs(float(np.sum(p)) - 1.0) > 1e-3:
            return False
        if not (sp.T_bounds[0] - 1e-6 <= T <= sp.T_bounds[1] + 1e-6):
            return False
        if not (sp.tau_bounds[0] - 1e-6 <= tau <= sp.tau_bounds[1] + 1e-6):
            return False

        proto = sp.protocols[int(k)]
        if proto.G is not None and proto.G.size > 0:
            if np.any(proto.G @ u - proto.h > 1e-5):
                return False
        return True

    def step(self, z: np.ndarray):
        k_vec, u_mat, proj_info = self.projector.project(z, enabled_protocols=self.enabled_protocols)

        violated = 0
        for j in range(self.spec.batch_size):
            if not self._check_feasible(int(k_vec[j]), u_mat[j]):
                violated += 1

        self._violations += violated
        self._steps += self.spec.batch_size

        obs, reward, terminated, truncated, info = self.base_env.step({"k": k_vec, "u": u_mat})
        info = dict(info)
        info["cvr_step"] = violated / max(1, self.spec.batch_size)
        info["cvr_running"] = self._violations / max(1, self._steps)
        info["proj_slots"] = proj_info.get("slots", None)
        return obs, reward, terminated, truncated, info


# -----------------------------
# 5) Minimal smoke test
# -----------------------------
def smoke_test():
    spec = make_protein_crystallization_spec(seed=0, batch_size=2, horizon=25)
    base = ProteinCrystallizationBaseEnv(spec)
    proj = HybridLogicProjector(spec)
    env = ProjectedProteinCrystallizationEnv(base, proj)

    obs, _ = env.reset(seed=123)
    ep_ret = 0.0
    for _ in range(spec.horizon):
        z = np.random.normal(size=env.action_space.shape[0]).astype(np.float32)
        obs, r, term, trunc, info = env.step(z)
        ep_ret += r
        if term or trunc:
            break

    print("Episode return:", ep_ret)
    print("Final best_quality:", info.get("best_quality"))
    print("Running CVR (should be ~0):", info.get("cvr_running"))


if __name__ == "__main__":
    smoke_test()
