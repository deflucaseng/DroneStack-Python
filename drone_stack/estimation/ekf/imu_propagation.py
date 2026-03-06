"""IMU state propagation and Jacobians — JAX implementation.

All public functions are JIT-compiled pure functions.  They accept and
return JAX arrays; the caller (EKF) is responsible for managing mutable
state on top.

State layout (15-dim):
    [p(0:3), v(3:6), q(6:10), bg(10:13), ba(13:16)]

Convention:
    - NED frame, gravity = [0, 0, 9.81]
    - Quaternion order: [w, x, y, z]
"""

import jax
import jax.numpy as jnp
from functools import partial

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G_NED = jnp.array([0.0, 0.0, 9.81])


# ---------------------------------------------------------------------------
# Quaternion / rotation helpers (pure-JAX, JIT-friendly)
# ---------------------------------------------------------------------------
@jax.jit
def quaternion_to_rotation_matrix(q: jnp.ndarray) -> jnp.ndarray:
    """Unit quaternion [w,x,y,z] → 3×3 rotation matrix."""
    qw, qx, qy, qz = q
    return jnp.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ])


@jax.jit
def normalize_quaternion(q: jnp.ndarray) -> jnp.ndarray:
    """Normalise quaternion and enforce positive scalar part."""
    q = q / jnp.linalg.norm(q)
    return jnp.where(q[0] < 0, -q, q)


# ---------------------------------------------------------------------------
# Matrix building blocks
# ---------------------------------------------------------------------------
@jax.jit
def omega_matrix(w: jnp.ndarray) -> jnp.ndarray:
    """4×4 Omega matrix: q_dot = 0.5 * Omega(w) @ q."""
    wx, wy, wz = w
    return jnp.array([
        [ 0.0, -wx, -wy, -wz],
        [  wx, 0.0,  wz, -wy],
        [  wy, -wz, 0.0,  wx],
        [  wz,  wy, -wx, 0.0],
    ])


@jax.jit
def xi_matrix(q: jnp.ndarray) -> jnp.ndarray:
    """4×3 Xi matrix: q_dot = 0.5 * Xi(q) @ w."""
    qw, qx, qy, qz = q
    return jnp.array([
        [-qx, -qy, -qz],
        [ qw, -qz,  qy],
        [ qz,  qw, -qx],
        [-qy,  qx,  qw],
    ])


@jax.jit
def dRa_dq(q: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    """3×4 Jacobian  ∂(R(q) @ a) / ∂q."""
    qw, qx, qy, qz = q
    ax, ay, az = a
    return 2.0 * jnp.array([
        [ qw*ax - qz*ay + qy*az,  qx*ax + qy*ay + qz*az, -qy*ax + qx*ay + qw*az, -qz*ax - qw*ay + qx*az],
        [ qz*ax + qw*ay - qx*az,  qy*ax - qx*ay - qw*az,  qx*ax + qy*ay + qz*az,  qw*ax - qz*ay + qy*az],
        [-qy*ax + qx*ay + qw*az,  qz*ax + qw*ay - qx*az, -qw*ax + qz*ay - qy*az,  qx*ax + qy*ay + qz*az],
    ])


# ---------------------------------------------------------------------------
# State propagation
# ---------------------------------------------------------------------------
@jax.jit
def propagate_state(
    x: jnp.ndarray,
    gyro: jnp.ndarray,
    accel: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Nonlinear IMU state propagation (first-order Euler).

    Args:
        x:     State vector (15,)
        gyro:  Raw gyroscope reading (3,) [rad/s]
        accel: Raw accelerometer reading (3,) [m/s²]
        dt:    Time step [s]

    Returns:
        Propagated state vector (15,)
    """
    p  = x[0:3]
    v  = x[3:6]
    q  = x[6:10]
    bg = x[10:13]
    ba = x[13:16]

    w_c = gyro  - bg
    a_c = accel - ba

    R = quaternion_to_rotation_matrix(q)

    q_new = normalize_quaternion(q + 0.5 * omega_matrix(w_c) @ q * dt)
    v_new = v + (R @ a_c + G_NED) * dt
    p_new = p + v * dt

    return jnp.concatenate([p_new, v_new, q_new, bg, ba])


# ---------------------------------------------------------------------------
# Analytical state-transition Jacobian
# ---------------------------------------------------------------------------
@jax.jit
def state_jacobian(
    x: jnp.ndarray,
    gyro: jnp.ndarray,
    accel: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Discrete-time state Jacobian  F ≈ I + Fc·dt.

    Evaluate at the *prior* state (call before propagate_state).
    """
    q  = x[6:10]
    bg = x[10:13]
    ba = x[13:16]

    w_c = gyro  - bg
    a_c = accel - ba

    R = quaternion_to_rotation_matrix(q)

    Fc = jnp.zeros((16, 16))

    # dp/dt = v
    Fc = Fc.at[0:3, 3:6].set(jnp.eye(3))

    # dv/dt  → dependence on q and ba
    Fc = Fc.at[3:6, 6:10].set(dRa_dq(q, a_c))
    Fc = Fc.at[3:6, 13:16].set(-R)

    # dq/dt  → dependence on q and bg
    Fc = Fc.at[6:10, 6:10].set(0.5 * omega_matrix(w_c))
    Fc = Fc.at[6:10, 10:13].set(-0.5 * xi_matrix(q))

    return jnp.eye(16) + Fc * dt