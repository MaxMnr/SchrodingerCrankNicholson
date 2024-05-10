import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.integrate import simps

def animRealImag(schrodinger):
    Nt = schrodinger.psi.shape[1]
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), tight_layout=True)
    fig.suptitle(r"Time evolution of $Re(\Psi)$ and $Im(\Psi)$")

    for ax in axs[-1]:
        ax.set_xlabel(r'Position $x$ [a.u.]')
    for ax in axs[:, 0]:
        ax.set_ylabel('Amplitude')
    
    axs = axs.flatten() 
    
    for ax in axs:
        ax.set_xlim((schrodinger.x[0], schrodinger.x[-1]))
        ax.set_ylim((-1.1 * np.max(abs(schrodinger.psi)), 1.1 * np.max(abs(schrodinger.psi))))
        ax.grid()
        ax.plot(schrodinger.x, schrodinger.V, c='gray', alpha=0.7)

    line_real = axs[1].plot([], [], c='#1f77b4', label=r'Re$(\Psi(x, t))$')[0]
    line_imag = axs[3].plot([], [], c='#ff7f0e', label=r'Im$(\Psi(x, t))$')[0]
    line_mod_1 = axs[1].plot([], [], 'g-', label=r'$|\Psi(x, t)|^2$')[0]
    line_mod_2 = axs[3].plot([], [], 'g-', label=r'$|\Psi(x, t)|^2$')[0]

    def update(i):
        line_real.set_data(schrodinger.x, np.real(schrodinger.psi[:, i]))
        line_imag.set_data(schrodinger.x, np.imag(schrodinger.psi[:, i]))
        line_mod_1.set_data(schrodinger.x, np.abs(schrodinger.psi[:, i]) ** 2)
        line_mod_2.set_data(schrodinger.x, np.abs(schrodinger.psi[:, i]) ** 2)
        return line_real, line_imag, line_mod_1, line_mod_2

    anim = FuncAnimation(fig, update, frames=range(Nt), blit=True, interval=10)

    # Plot initial wavefunction at t=0
    axs[0].plot(schrodinger.x, np.real(schrodinger.psi[:, 0]), c='#1f77b4', label=r'Re$(\Psi(x, t=0))$')
    axs[2].plot(schrodinger.x, np.imag(schrodinger.psi[:, 0]), c='#ff7f0e', label=r'Im$(\Psi(x, t=0))$')

    for ax in axs:
        ax.legend()

    return anim


def animModulus(schrodinger):
    Nt = schrodinger.psi.shape[1]

    fig, ax = plt.subplots()

    ax.set_xlabel("Position")
    ax.set_ylabel(r"$|\Psi|^2$")
    ax.set_xlim(min(schrodinger.x), max(schrodinger.x))
    ax.set_ylim(np.min(np.abs(schrodinger.psi)**2)/2, np.max(np.abs(schrodinger.psi)**2)/2)
    
    ax.plot(schrodinger.x, schrodinger.V, "r", label=r"$V$")
    
    lines, = ax.plot([], [], lw=1, color="black", label=r"$|\Psi|^2$")
    
    def update(frame):
        lines.set_data(schrodinger.x, np.abs(schrodinger.psi[:, frame])**2)
        return lines,

    anim = FuncAnimation(fig, update, frames=range(Nt), blit=True, interval=1)
    
    txt = r"$\kappa=$" + str(schrodinger.kappa) + "\n" + r"$\sigma=$" + str(schrodinger.sigma)
    ax.legend(title=txt)

    return anim


def plotUncertainty(schrodinger):
    mean_x = np.zeros(schrodinger.Nt)
    mean_x_2 = np.zeros(schrodinger.Nt)
    
    for i in range(schrodinger.Nt):
        x = schrodinger.x
        y = schrodinger.psi[:, i]
        mean_x[i] = simps(x * np.abs(y) ** 2, x=x)
        mean_x_2[i] = simps(x**2 * np.abs(y) ** 2, x=x)

    plt.plot(schrodinger.t, mean_x_2 - mean_x**2, lw=1, c='black')

    plt.xlabel(r"Time $t$ [u.a.]")
    plt.ylabel(r"$\langle x^2 \rangle - \langle x \rangle^2$")
    plt.title("Uncertainty")
    plt.grid(True)

def plotNormalization(schrodinger):
    norm = np.zeros(schrodinger.Nt)
    for i in range(schrodinger.Nt):
        norm[i] = np.trapz(np.abs(schrodinger.psi[:, i]) ** 2, x=schrodinger.x)
    plt.plot(schrodinger.t, norm, c="black", lw=1)
    plt.grid()
    plt.xlabel("Time [a.u.]")
    plt.ylabel(r"$\int |\Psi|^2 dx$")

def plotExpectedPosition(schrodinger, ret=False):
    plt.title("Expected Position [u.a.]")
    plt.xlabel("Time [u.a.]")
    plt.ylabel(r"$\langle x \rangle$")
    
    expected = np.zeros(schrodinger.Nt)
    for i in range(schrodinger.Nt):
        y = schrodinger.psi[:, i]
        expected[i] = np.trapz(schrodinger.x * np.abs(y) ** 2, x=schrodinger.x)
    if ret == False:
        plt.plot(schrodinger.t, expected, c="black", lw=1)
    else:
        return expected
    plt.grid(True)

def plotTimeEvolution(schrodinger):
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(schrodinger.x, schrodinger.t, np.abs(schrodinger.psi).T, cmap='gray')
    fig.colorbar(
        mesh,
        ax=ax,
        orientation='vertical',
        label=r'$|\psi(x)|$',
        fraction=0.06, pad=0.02,
    )
    plt.ylabel("Time [u.a.]")
    plt.xlabel("Position [u.a.]")
    plt.title("Time Evolution of the packet")
    

def plotSome(schrodinger):
    fig, axs = plt.subplots(ncols=1, nrows=5, sharex=True, sharey=True, figsize=(9, 7))
    t_ = 0
    dt_ = schrodinger.Nt // axs.size - 1
    for ax in axs.flatten():
        ax.set_title(f'Time: {t_/schrodinger.Nt} [u.a.]')
        ax.plot(schrodinger.x, np.real(schrodinger.psi[:, t_]), label=r"Re($\Psi$)")
        ax.plot(schrodinger.x, np.imag(schrodinger.psi[:, t_]), label=r"Im($\Psi$)")
        ax.grid(True)
        t_ += dt_
        ax.legend()
    plt.tight_layout()
    
