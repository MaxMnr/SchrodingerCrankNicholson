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
    kappa, sigma = (500/L, L/20)
    tmax = 100
    #==========================================================

    # Here the initial wave packet is seen as a superposition of two gaussian packet
    schrodinger = solver.Schrodinger(Nx, Nt, kappa, sigma, L, x0, potential="tunneling", tmax=tmax)

    # Solve the equation using Crank-Nicholson Scheme 
    schrodinger.solve()


    #================ Animate the evolution of the packet (Re and Im)=====
    #anim = utils.animRealImag(schrodinger)
    #=====================================================================

    #================ Animate the evolution of the packet (Modulus)=========
    #anim = utils.animModulus(schrodinger)
    #=======================================================================
    
    #writer = animation.FFMpegWriter(fps=60)
    #anim.save('anim2.mp4', writer=writer)
    #print("done")

    #==================== plot the Expected Position ====================
    utils.plotExpectedPosition(schrodinger)
    #=======================================================================

    #==================== plot Some ====================
    utils.plotSome(schrodinger)
    #=======================================================================

    #==================== plot time evolution ====================
    utils.plotTimeEvolution(schrodinger)
    #=======================================================================


    plt.show()
