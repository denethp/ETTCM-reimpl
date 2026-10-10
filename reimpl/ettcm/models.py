"""Global motion models: the B matrix of Eq. (9) and its reduced versions (Sec. 4.2).

Convention used in THIS re-implementation = the paper's convention:
    image velocity of event i:   v_i = B_i theta,   B_i = ( rho_i V_i | Omega_i )
    theta = (nu, omega) = camera linear / angular velocity
    divergence mu_i = rho_i * nu_z,   TTC tau_i = 1 / mu_i                 (Eq. 1)

(The official C++ code uses B_code = -B_paper, hence theta_code = -theta_paper and
 its divergence is -rho * theta_code_z. Same estimator, opposite sign.)

p = (x - cx, y - cy) is the rectified pixel position relative to the principal
point (assumed to be the focus of expansion).

Model ids (keep them as ints so Numba can branch on them):
    SCALING        theta = (nu_z,)                         n_vars = 1, n_v = 1
    TRANSLATION    theta = (nu_x, nu_y, nu_z)              n_vars = 3, n_v = 3
    SIX_DOF        theta = (nu_x, nu_y, nu_z, wx, wy, wz)  n_vars = 6, n_v = 3
    SIX_DOF_PARAM  same output theta, but estimated with the official code's
                   re-parameterised 6-DOF model (ParameterizedSixDOF), which is
                   what the paper's Table 2 6-DOF rows were computed with.
                   See estimator._run_param.

The "translational part" of the parameters is theta[:n_v]; nu_z is theta[n_v - 1].
"""
from ._jit import njit

SCALING = 0
TRANSLATION = 1
SIX_DOF = 2
SIX_DOF_PARAM = 3

MODEL_IDS = {"scaling": SCALING, "translation": TRANSLATION, "six_dof": SIX_DOF,
             "six_dof_param": SIX_DOF_PARAM}


@njit
def n_vars(model):
    if model == SCALING:
        return 1
    if model == TRANSLATION:
        return 3
    return 6  # SIX_DOF and SIX_DOF_PARAM


@njit
def n_v(model):
    if model == SCALING:
        return 1
    return 3


@njit
def b_matrix(model, px, py, fx, fy, rho, B):
    """Fill B (shape (2, >= n_vars)) for a point p = (px, py) and inverse depth rho.

    SCALING:      B = rho * (px, py)^T                       (only nu_z)
    TRANSLATION:  B = rho * V_i,  V_i = [[-fx, 0, px], [0, -fy, py]]
    SIX_DOF:      B = (rho * V_i | Omega_i), Omega_i from Eq. (9)
    """
    if model == SCALING:
        B[0, 0] = rho * px
        B[1, 0] = rho * py
        return

    # Translational part: rho * V_i
    B[0, 0] = -rho * fx
    B[0, 1] = 0.0
    B[0, 2] = rho * px
    B[1, 0] = 0.0
    B[1, 1] = -rho * fy
    B[1, 2] = rho * py
    if model == SIX_DOF:
        # Rotational part Omega_i (does not depend on depth)
        B[0, 3] = px * py / fy
        B[0, 4] = -fx - px * px / fx
        B[0, 5] = py * fx / fy
        B[1, 3] = fy + py * py / fy
        B[1, 4] = -px * py / fx
        B[1, 5] = -px * fy / fx


# ---------------------------------------------------------------------------
# Official ParameterizedSixDOF (include/ETTCM/motion_model.hpp), CODE sign
# convention (B_code = -B_paper). Unknowns m = (m0..m5) with
#     m0 = rho*nu_x + w_y,  m1 = rho*nu_y - w_x,  m2 = nu_z,  (m3, m4, m5) = w
# (all in code sign), so the linear velocity is
#     v3 = ((m0 - m4) / rho, (m1 + m3) / rho, m2).
# ---------------------------------------------------------------------------
@njit
def b_matrix_param(px, py, fx, fy, rho, B):
    """b_matrix_param(p, f, rho): image velocity = B @ m."""
    B[0, 0] = fx
    B[0, 1] = 0.0
    B[0, 2] = -rho * px
    B[0, 3] = -px * py / fy
    B[0, 4] = px * px / fx
    B[0, 5] = -py * fx / fy
    B[1, 0] = 0.0
    B[1, 1] = fy
    B[1, 2] = -rho * py
    B[1, 3] = -py * py / fy
    B[1, 4] = px * py / fx
    B[1, 5] = px * fy / fx


@njit
def b_matrix_code6(px, py, fx, fy, rho, B):
    """Official SixDOF/ParameterizedSixDOF b_matrix(p, f, rho) = -B_paper."""
    B[0, 0] = rho * fx
    B[0, 1] = 0.0
    B[0, 2] = -rho * px
    B[0, 3] = -px * py / fy
    B[0, 4] = fx + px * px / fx
    B[0, 5] = -py * fx / fy
    B[1, 0] = 0.0
    B[1, 1] = rho * fy
    B[1, 2] = -rho * py
    B[1, 3] = -(fy + py * py / fy)
    B[1, 4] = px * py / fx
    B[1, 5] = px * fy / fx
