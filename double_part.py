import solver
import utils
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"
})


if __name__ == "__main__":
    #========== Defining the relevant parameters ==============
    L = 100
    Nx, Nt = (4*L, 10000)
    x0 = -L/3
    x1 = L/3
    kappa, sigma = (500/L, L/20)
    tmax = 40
    #==========================================================

    # Here the initial wave packet is seen as a superposition of two gaussian packet
    schrodinger = solver.Schrodinger(Nx, Nt, kappa, sigma, L, x0, potential="ISW", tmax=tmax)
    schrodinger.psi0 += schrodinger.gaussianPacket(schrodinger.x, x1, -schrodinger.kappa, schrodinger.sigma*0.8, norm=True)
    # Solve the equation using Crank-Nicholson Scheme 
    schrodinger.solve()


    #================ Animate the evolution of the packet (Re and Im)=====
    #anim = utils.animRealImag(schrodinger)
    #=====================================================================

    #================ Animate the evolution of the packet (Modulus)=========
    anim = utils.animModulus(schrodinger)
    #=======================================================================

    #==================== plot the Expected Position ====================
    utils.plotExpectedPosition(schrodinger)
    #plt.savefig("images/double_part/expected_plot_double_hr.png", dpi=1080, transparent=True)
    #=======================================================================

    #==================== plot Some ====================
    utils.plotSome(schrodinger)
    #plt.savefig("images/double_part/some_plot_double_hr.png", dpi=1080, transparent=True)
    #=======================================================================

    #==================== plot time evolution ====================
    utils.plotTimeEvolution(schrodinger)
    #plt.savefig("images/double_part/time_evol_double_hr.png", dpi=1080, transparent=True)
    #=======================================================================
    plt.figure()
    utils.plotUncertainty(schrodinger)

    plt.show()
