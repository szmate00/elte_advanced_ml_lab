import autograd.numpy as np
from autograd import grad
from scipy.integrate import solve_ivp


def hamiltonian(qp):
    q = qp[0]
    p = qp[1]
    return 0.5 * p**2 + 0.5 * q**2

def symplectic(t, qp):
    nabla_h = grad(hamiltonian)(qp)
    dhdq = nabla_h[0]
    dhdp = nabla_h[1]
    return [dhdp, -dhdq]    

def mass_spring(energies, t_span, t_points):
    q = []
    p = []

    q_dot = []
    p_dot = []

    t_eval = np.linspace(t_span[0], t_span[1], t_points)

    for E in energies:
        q_init = np.sqrt(2 * E) * np.cos(t)
        p_init = np.sqrt(2 * E) * np.sin(t)
        y0 = [q_init, p_init]
        
        symplectic_ivp = solve_ivp(fun=symplectic, t_span=t_span, y0=y0, t_eval=t_eval)
        q.append(symplectic_ivp['y'][0] + np.random.normal(0, 0.1, t_span[1]))
        p.append(symplectic_ivp['y'][1] + np.random.normal(0, 0.1, t_span[1]))
        
        dots = np.array([np.array(symplectic(None, qp)) for qp in symplectic_ivp['y'].T])
        q_dot.append(dots[:,0])
        p_dot.append(dots[:,1])
            
    q = np.array(q)
    p = np.array(p)

    q_dot = np.array(q_dot)
    p_dot = np.array(p_dot)

    return q, p, q_dot, p_dot