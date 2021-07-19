'''Dispersion relation for the symmetry breaking case.
Author: Huaiyu Duan (UNM)
Ref: arXiv:1901.01546
'''

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root, root_scalar, minimize_scalar

def _DR_func(K, Omega, G, opts={}):
    '''The function D(K, Omega) that should equal 0. See Eq.35 of Ref. 
    K : wave number
    Omega : wave frequency
    G(u) : ELN distribution function
    opts : dictionary of options to be passed on to the integrator

    return : value of the function
    '''
    kr, ki = K.real, K.imag
    wr, wi = Omega.real, Omega.imag
    fr = lambda u: G(u)*(wr-kr*u)* (1-u*u)/((wr-kr*u)**2 + (wi-ki*u)**2)
    fi = lambda u: G(u)*(-wi+ki*u)* (1-u*u)/((wr-kr*u)**2 + (wi-ki*u)**2)
    r, *_ = quad(fr, -1, 1, **opts)
    i, *_ = quad(fi, -1, 1, **opts)

    return complex(r, i) - 2

def _Omega_of_real_n(n, G, opts={}):
    '''Calculate the frequency of the real dispersion relation. See Eq.43a of Ref.
    n : refractive index
    G(u) : ELN distribution function for u in [-1, 1]
    opts : dictionary of options to be passed on to the integrator

    return : collective wave frequency
    '''
    f = lambda u: G(u)*(1 - u*u)/(1-n*u)
    res, *_ = quad(f, -1, 1, **opts)
    return 0.5*res


def _dOmega_dn(n, G, opts={}):
    '''Calculate dOmega/dn of the real dispersion relation.
    n : refractive index
    G(u) : ELN distribution function for u in [-1, 1]
    opts : dictionary of options to be passed on to the integrator

    return : dOmega/dn
    '''
    f = lambda u: G(u)*(1. - u*u)*u/(1.-n*u)**2
    res, *_ = quad(f, -1, 1, **opts)
    return 0.5*res


def _dK_dn(n, G, opts={}):
    '''Calculate dK/dn = Omega + n dOmega/dn of the real dispersion relation.
    n : refractive index
    G(u) : ELN distribution function for u in [-1, 1]
    opts : dictionary of options to be passed on to the integrator

    return : dK/dn
    '''
    f = lambda u: G(u)*(1. - u*u)/(1.-n*u)**2
    res, *_ = quad(f, -1, 1, **opts)
    return 0.5*res


def _extremalOmega(G, opts={}, eps=1e-5):
    '''Find the extremal points of Omega(K) on the real dispersion relation.
    G(u) : ELN distribution function for u in [-1, 1]
    opts : dictionary of options to be passed on to the integrator
    eps : numerical error tolerance in n

    return : [(n, K, Omega), ...], where n, K, and Omega are the values of refractive indices, wave numbers, and frequencies at the extremal points.
    '''
    assert G(-1) > 0, "G(-1) should be positive."
    Omega = lambda n: _Omega_of_real_n(n, G, opts=opts)
    dOmega_dn = lambda n: _dOmega_dn(n, G, opts=opts)

    if G(1) >= 0: # no crossing, one minimum point
        sol1 = root_scalar(dOmega_dn, bracket=(-1+eps, 1-eps))
        assert sol1.converged, "Couldn't find an extremal Omega point."
        n1 = sol1.root
        w1 = Omega(n1)
        return [n1, n1*w1, w1]
    else: # crossing
        sol = minimize_scalar(lambda n: -dOmega_dn(n), bounds=(-1+eps, 1-eps), method="bounded") 
        assert sol.success, "Couldn't find the extremal Omega points."
        n1 = sol.x
        w1 = Omega(n1)
        val = dOmega_dn(n1)
        if abs(val) <= eps*abs(w1): # degenerate extremal points
            return [ [n1, n1], [n1*w1, n1*w1], [w1, w1] ]
        elif val > 0: # two distinct extremal points
            sol1 = root_scalar(dOmega_dn, bracket=(-1+eps, n1)) # find the minimum Omega point
            sol2 = root_scalar(dOmega_dn, bracket=(n1, 1-eps)) # find the maximum Omega point
            assert sol1.converged and sol2.converged, "Couldn't find both extremal points."
            n1 = sol1.root; n2 = sol2.root # values where Omega are extreme
            w1 = Omega(n1); w2 = Omega(n2)
            return [ [n1, n2], [n1*w1, n2*w2], [w1, w2] ]

    # no extremal Omega points found
    return [None, None, None]
    

def _extremalK(G, opts={}, eps=1e-5):
    '''Find the extremal points of K(Omega) on the real dispersion relation.
    G(u) : ELN distribution function for u in [-1, 1]
    opts : dictionary of options to be passed on to the integrator
    eps : a small number to guard against rare cases

    return : [n, K, Omega], where n, K, and Omega are the values of refractive indices, wave numbers, and frequencies at the extremal points.
    '''
    assert G(-1) > 0, "G(-1) should be positive."
    if G(1) >= 0: # no crossing, no extremal point
        return [None, None, None]
  
    dK_dn = lambda n: _dK_dn(n, G, opts=opts)
    sol1 = root_scalar(dK_dn, bracket=(-1+eps, 1-eps))
    assert sol1.converged, "Couldn't find the extremal K point. The crossing may be too shallow."
    n1 = sol1.root
    w1 = _Omega_of_real_n(n1, G, opts=opts)
    return [n1, n1*w1, w1]


def _xing(G, opts={}):
    '''Find the crossing point of the ELN and the corresponding critical point in (K, Omega).
    G(u) : ELN distribution function for u in [-1, 1]
    opts : dictionary of options to be passed on to the integrator

    return : [n, K, Omega], where n, K, and Omega are the values of refractive indices, wave numbers, and frequencies at the critical point.
    '''
    assert G(-1) > 0, "G(-1) should be positive."
    Omega = lambda n: _Omega_of_real_n(n, G, opts=opts)
    if G(1) >= 0: # no crossing
        return [None, None, None]

    sol = root_scalar(G, bracket=(-1, 1))
    assert sol.converged, "Couldn't find the crossing point."
    I, *_ = quad(lambda u: G(u)*(1-u*u), -1, 1, weight='cauchy', wvar=sol.root, **opts)
    n = 1 / sol.root
    k = -0.5 * I
    return [n, k, k/n]


def _cplx_K_of_0(G, opts={}):
    '''Find the complex K with at Omega=0.
    G(u) : ELN distribution function for u in [-1, 1]
    opts : dictionary of options to be passed on to the integrator

    return : K
    '''
    res, *_ = quad(lambda u: G(u)*(1-u*u), -1, 1, weight='cauchy', wvar=0, **opts)
    return complex(-0.5*res, 0.5*np.pi*abs(G(0)))


def _cplx_K_of_Omega(Omega, G, K0=0.1j, opts={}):
    '''Find the complex K with given real Omega.
    Omega : wave frequency
    G(u) : ELN distribution function for u in [-1, 1]
    K0 : initial guess of K
    opts : dictionary of options to be passed on to the integrator

    return : K
    '''
    def eq(k): # equation for solving Omega
        res = _DR_func(complex(k[0], k[1]), Omega, G, opts)
        return res.real, res.imag
    sol = root(eq, x0=[K0.real, K0.imag])
    assert sol.success, f"{sol.message} Couldn't find a complex K solution at Omega={Omega}."
    return complex(sol.x[0], abs(sol.x[1]))


def _cplx_K(Omega, G, K0=0.1j, opts={}):
    '''Find the complex K with given list of real Omegas.
    Omega[num] : NumPy array of wave frequencies
    G(u) : ELN distribution function for u in [-1, 1]
    K0 : initial guess of K
    opts : dictionary of options to be passed on to the integrator

    return : K[num]
    '''
    kk = np.empty(len(Omega), dtype=np.cdouble)
    for i in range(len(Omega)):
        kk[i] = K0 = _cplx_K_of_Omega(Omega[i], G, K0, opts=opts)
    return kk


def _cplx_Omega_of_K(K, G, Omega0=0.1j, opts={}):
    '''Find the complex Omega with given real K.
    K : wave number
    G(u) : ELN distribution function for u in [-1, 1]
    Omega0 : guess value of Omega
    opts : dictionary of options to be passed on to the integrator

    return : Omega
    '''
    def eq(w): # equation for solving Omega
        res = _DR_func(K, complex(w[0], w[1]), G, opts)
        return res.real, res.imag
    sol = root(eq, x0=[Omega0.real, Omega0.imag])
    assert sol.success, f"{sol.message} Couldn't find a complex Omega solution at K={K}."
    return complex(sol.x[0], abs(sol.x[1]))


def _cplx_Omega(K, G, Omega0=0.1j, opts={}):
    '''Find the complex Omega with given list of real K.
    K[num] : NumPy array of wave numbers
    G(u) : ELN distribution function for u in [-1, 1]
    Omega0 : guess value of Omega
    opts : dictionary of options to be passed on to the integrator

    return : Omega[num]
    '''
    ww = np.empty(len(K), dtype=np.cdouble)
    for i in range(len(K)):
        ww[i] = Omega0 = _cplx_Omega_of_K(K[i], G, Omega0, opts=opts)
    return ww

def DR_real(G, num_pts=100, opts={}, shift=True):
    '''Calculate the real dispersion relation of the fast oscillation wave.
    G(u) : ELN distribution function for u in [-1, 1]
    num_pts : number of points to calculate
    opts : dictionary of options to be passed on to the integrator
    shift : whether to shift the frequency and wave number so that the forbidden region centers at (0, 0)
    
    return : [(K[num_pts], Omega[num_pts])] with K and Omega being the NumPy arrays of the wave numbers and frequencies
    '''
    assert G(-1) > 0, "G(-1) should be positive."
    Omega = lambda n: _Omega_of_real_n(n, G, opts=opts)
    nn = np.linspace(-1, 1, num_pts) # list of refractive indices
    kk = np.empty(num_pts, dtype=np.double) # list of wave numbers
    ww = np.empty(num_pts, dtype=np.double) # list of the frequencies
    for i in range(num_pts): # compute the DR
        ww[i] = Omega(nn[i])
        kk[i] = nn[i] * ww[i]

    if not shift: # shift back Omega and K if needed
        w0, *_ = quad(G, -1, 1, **opts) # shift of the frequency
        k0, *_ = quad(lambda u: G(u)*u, -1, 1, **opts) # shift of the wave number
        kk += k0
        ww += w0

    return [(kk, ww)]


def DR_complexK(G, num_pts=100, opts={}, shift=True, eps=1e-5):
    '''Calculate the complex K dispersion relation of the fast oscillation wave.
    G(u) : ELN distribution function for u in [-1, 1]
    num_pts : number of points to calculate
    opts : dictionary of options to be passed on to the integrator
    shift : whether to shift the frequency and wave number so that the forbidden region centers at (0, 0)
    eps : numerical error tolerance in n
    
    return : [(K[num_pts], Omega[num_pts]), ...] with K and Omega being the NumPy arrays of the wave numbers and frequencies
    '''
    assert G(-1) > 0, "G(-1) should be positive."
    dr = [] # dispersion relations

    nc, kc, wc = _extremalOmega(G, opts=opts, eps=eps) # extremal Omega points on the real branch
    k0 = _cplx_K_of_0(G, opts=opts) # complex K at Omega = 0
    if G(1) >= 0: # no crossing, one complex K branch
        ww = np.linspace(wc, 0, num_pts)
        kk = np.empty(num_pts, dtype=np.cdouble)
        kk[0] = kc; kk[-1] = k0
        kk[1:-1] = _cplx_K(ww[1:-1], G, kk[0]*(1+0.1j), opts=opts)
        dr.append((kk, ww))
    else: # crossing
        nx, kx, wx = _xing(G, opts=opts) # critical points related to the crossing point
        if nc == None: # deep crossing, one complex K branch
            ww = np.linspace(wx, 0, num_pts)
            kk = np.empty(num_pts, dtype=np.cdouble)
            kk[0] = kx; kk[-1] = k0
            kk[1:-1] = _cplx_K(ww[1:-1], G, kk[0]*(1+0.1j), opts=opts)
            dr.append((kk, ww))
        else: # moderate crossing, two seperate complex K branches
            # first branch
            ww = np.linspace(wc[0], 0, num_pts)
            kk = np.empty(num_pts, dtype=np.cdouble)
            kk[0] = kc[0]; kk[-1] = k0
            kk[1:-1] = _cplx_K(ww[1:-1], G, kk[0]*(1+0.1j), opts=opts)
            dr.append((kk, ww))
            # second branch
            ww = np.linspace(wc[1], wx, num_pts)
            kk = np.empty(num_pts, dtype=np.cdouble)
            kk[0] = kc[1]; kk[-1] = kx
            kk[1:-1] = _cplx_K(ww[1:-1], G, kk[0]*(1+0.1j), opts=opts)
            dr.append((kk, ww))

    if not shift: # shift back Omega and K if needed
        w0, *_ = quad(G, -1, 1, **opts) # shift of the frequency
        k0, *_ = quad(lambda u: G(u)*u, -1, 1, **opts) # shift of the wave number
        for kk, ww in dr:
            kk += k0
            ww += w0
    return dr

def DR_complexOmega(G, num_pts=100, opts={}, shift=True, eps=1e-5):
    '''Calculate the complex Omega dispersion relation of the fast oscillation wave.
    G(u) : ELN distribution function for u in [-1, 1]
    num_pts : number of points to calculate
    opts : dictionary of options to be passed on to the integrator
    shift : whether to shift the frequency and wave number so that the forbidden region centers at (0, 0)
    eps : numerical error tolerance in n

    return : [(K[num_pts], Omega[num_pts])] with K and Omega being the NumPy arrays of the wave numbers and frequencies
    '''
    assert G(-1) > 0, "G(-1) should be positive."
    dr = [] # dispersion relations
    if G(1) < 0: # crossing, one complex Omega branch
        nc, kc, wc = _extremalK(G, opts=opts, eps=eps)
        nx, kx, wx = _xing(G, opts=opts)
        kk = np.linspace(kc, kx, num_pts)
        ww = np.empty(num_pts, dtype=np.cdouble)
        ww[0] = wc; ww[-1] = wx
        ww[1:-1] = _cplx_Omega(kk[1:-1], G, kk[0]*(1+0.1j), opts=opts)
        dr.append((kk, ww))

    if not shift: # shift back Omega and K if needed
        w0, *_ = quad(G, -1, 1, **opts) # shift of the frequency
        k0, *_ = quad(lambda u: G(u)*u, -1, 1, **opts) # shift of the wave number
        for kk, ww in dr:
            kk += k0
            ww += w0
    return dr


