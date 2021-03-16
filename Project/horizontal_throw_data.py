import autograd.numpy as np
from autograd import grad
from scipy.integrate import solve_ivp


def hamiltonian(qp):
    q_x = qp[0]
    q_y = qp[1]
    p_x = qp[2]
    p_y = qp[3]
    return (p_x**2 + p_y**2) / 2 - 9.81 * q_y

def symplectic(t, qp):
    nabla_h = grad(hamiltonian)(qp)
    dhdq = np.array([nabla_h[0], nabla_h[1]])
    dhdp = np.array([nabla_h[2], nabla_h[3]])
    return np.array([dhdp, -dhdq]).flatten()

def horizontal_throw(energies, t_span, t_points):
    q = []
    p = []

    q_dot = []
    p_dot = []

    t_eval = np.linspace(t_span[0], t_span[1], t_points)

    for E in energies:
        qy_init = 1
        qx_init = 0
        px_init = np.sqrt(2 * E + 2 * 9.81) * np.cos(t)
        py_init = np.sqrt(2 * E + 2 * 9.81) * np.sin(t)
        y0 = [[qx_init, qy_init], [px_init, py_init]]
        
        symplectic_ivp = solve_ivp(fun=symplectic, t_span=t_span, y0=y0, t_eval=t_eval)
        q.append(np.array([symplectic_ivp['y'][0] + np.random.normal(0, 0.1, t_span[1]), symplectic_ivp['y'][1] + np.random.normal(0, 0.1, t_span[1])]))
        p.append(np.array([symplectic_ivp['y'][2] + np.random.normal(0, 0.1, t_span[1]), symplectic_ivp['y'][3] + np.random.normal(0, 0.1, t_span[1])]))
        
        dots = np.array([np.array(symplectic(None, qp)) for qp in symplectic_ivp['y'].T])

        # dimension of these two arrays below need some work
        q_dot.append(np.array([dots[:,0], dots[:,1]]))
        p_dot.append(np.array([dots[:,2], dots[:,3]]))
            
    q = np.array(q)
    p = np.array(p)

    q_dot = np.array(q_dot)
    p_dot = np.array(p_dot)

    return q, p, q_dot, p_dot