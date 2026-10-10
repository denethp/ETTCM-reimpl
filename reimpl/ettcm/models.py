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
    SCALING     theta = (nu_z,)                         n_vars = 1, n_v = 1
    TRANSLATION theta = (nu_x, nu_y, nu_z)              n_vars = 3, n_v = 3
    SIX_DOF     theta = (nu_x, nu_y, nu_z, wx, wy, wz)  n_vars = 6, n_v = 3

The "translational part" of the parameters is theta[:n_v]; nu_z is theta[n_v - 1].
"""
from ._jit import njit

SCALING = 0
TRANSLATION = 1
SIX_DOF = 2

MODEL_IDS = {"scaling": SCALING, "translation": TRANSLATION, "six_dof": SIX_DOF}


@njit
def n_vars(model):
    if model == SCALING:
        return 1
    if model == TRANSLATION:
        return 3
    return 6


@njit
def n_v(model):
    if model == SCALING:
        return 1
    return 3


@njit
def b_matrix(model, px, py, fx, fy, rho, B):
    """Fill B (shape (2, >= n_vars)) for a point p = (px, py) and inverse depth rho.

    Worked example (done for you): SCALING.
        V_i third column is (px, py), so v = rho * (px, py) * nu_z.
    """
    if model == SCALING:
        B[0, 0] = rho * px
        B[1, 0] = rho * py
        return

    # ---------------------------------------------------------------- TODO (M2)
    # TRANSLATION:  B = rho * V_i  with
    #     V_i = [[-fx,   0, px],
    #            [  0, -fy, py]]
    # SIX_DOF: the same first 3 columns, plus Omega_i (Eq. 9) in columns 3..5:
    #     Omega_i = [[ px*py/fy,      -fx - px*px/fx,  py*fx/fy],
    #                [ fy + py*py/fy, -px*py/fx,      -px*fy/fx]]
    # Write the entries one by one (no np.array(...) inside njit hot loops).
    raise NotImplementedError("b_matrix: TRANSLATION / SIX_DOF not written yet")
