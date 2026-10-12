"""Global motion models used for the paper's Table 2: the B matrix of Eq. (9).

Convention used in THIS re-implementation = the paper's convention:
    image velocity of event i:   v_i = B_i theta,   B_i = ( rho_i V_i | Omega_i )
    theta = (nu, omega) = camera linear / angular velocity
    divergence mu_i = rho_i * nu_z,   TTC tau_i = 1 / mu_i                 (Eq. 1)

(The official C++ code uses B_code = -B_paper, hence theta_code = -theta_paper and
 its divergence is -rho * theta_code_z. Same estimator, opposite sign.)

p = (x - cx, y - cy) is the rectified pixel position relative to the principal
point (assumed to be the focus of expansion).

Model ids (ints so Numba can branch on them):
    SCALING      theta = (nu_z,)                         n_vars = 1, n_v = 1
    DRIVING      theta = (nu_z, w_y)                     n_vars = 2, n_v = 1
                 Sec. 4.2, used on MVSEC outdoor_day1 (a car); |nu_z| = 1 is
                 imposed, as in the official code's Driving model.
    TRANSLATION  theta = (nu_x, nu_y, nu_z)              n_vars = 3, n_v = 3
    SIX_DOF      theta = (nu_x, nu_y, nu_z, wx, wy, wz)  n_vars = 6, n_v = 3
                 Estimated exactly like the official code does it, with the
                 re-parameterised model ParameterizedSixDOF (see b_matrix_param
                 and estimator._run_param). This is what the paper's 6-DOF rows
                 were computed with; the paper itself does not describe it.

The "translational part" of the parameters is theta[:n_v]; nu_z is theta[n_v - 1].
"""
from ._jit import njit

SCALING = 0
TRANSLATION = 1
SIX_DOF = 2
DRIVING = 3

MODEL_IDS = {"scaling": SCALING, "translation": TRANSLATION, "six_dof": SIX_DOF,
             "driving": DRIVING}


@njit
def n_vars(model):
    if model == SCALING:
        return 1
    if model == TRANSLATION:
        return 3
    if model == DRIVING:
        return 2
    return 6


@njit
def n_v(model):
    if model == SCALING or model == DRIVING:
        return 1
    return 3


@njit
def b_matrix(model, px, py, fx, fy, rho, B):
    """Fill B (shape (2, >= n_vars)) for SCALING, TRANSLATION or DRIVING, paper convention.

    SCALING:      B = rho * (px, py)^T                    (only nu_z)
    TRANSLATION:  B = rho * V_i,  V_i = [[-fx, 0, px], [0, -fy, py]]
    DRIVING:      B = [[rho*px, -fx - px^2/fx], [rho*py, -px*py/fx]]   (nu_z, w_y)
    """
    if model == SCALING:
        B[0, 0] = rho * px
        B[1, 0] = rho * py
        return
    if model == DRIVING:
        B[0, 0] = rho * px
        B[1, 0] = rho * py
        B[0, 1] = -fx - px * px / fx
        B[1, 1] = -px * py / fx
        return
    B[0, 0] = -rho * fx
    B[0, 1] = 0.0
    B[0, 2] = rho * px
    B[1, 0] = 0.0
    B[1, 1] = -rho * fy
    B[1, 2] = rho * py


# ---------------------------------------------------------------------------
# 6-DOF: official ParameterizedSixDOF (include/ETTCM/motion_model.hpp), CODE sign
# convention (B_code = -B_paper). Unknowns m = (m0..m5) with
#     m0 = rho*nu_x + w_y,  m1 = rho*nu_y - w_x,  m2 = nu_z,  (m3, m4, m5) = w
# (all in code sign), so the linear velocity is
#     v3 = ((m0 - m4) / rho, (m1 + m3) / rho, m2).
# ---------------------------------------------------------------------------
@njit
def b_matrix_param(px, py, fx, fy, rho, B):
    """ParameterizedSixDOF::b_matrix_param(p, f, rho): image velocity = B @ m."""
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
    """ParameterizedSixDOF::b_matrix(p, f, rho) = -B_paper of Eq. (9).

    Used by the official code to build its Gauss-Newton Jacobian (see
    estimator._run_param)."""
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
