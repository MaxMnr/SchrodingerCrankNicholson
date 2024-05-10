import numpy as np 
import tqdm

class Schrodinger:
    def __init__(self, Nx, Nt, kappa, sigma, L, x0, potential="ISW", tmax=10):
        """
        Nx: Number of spatial grid points
        Nt: Number of time steps
        kappa, sigma: Parameters for Gaussian wave packet
        L: Length of the spatial domain
        x0: Initial position of the wave packet
        potential: Type of potential (default is Infinite Square Well)
        tmax: Maximum time for simulation
        """
        
        # Assign parameters to class attributes
        self.Nx, self.Nt = Nx, Nt
        self.x, self.dx = np.linspace(-L/2, L/2, self.Nx, retstep=True)  # Spatial grid and step size
        self.tmax = tmax  # Maximum time for simulation
        self.t, self.dt = np.linspace(0, self.tmax, self.Nt, retstep=True)  # Time grid and time step
        self.kappa, self.sigma = kappa, sigma  # Parameters for Gaussian wave packet
        self.L = L  # Length of the spatial domain
        self.x0 = x0  # Initial position of the wave packet

        # Initialize wave packet at t=0 using Gaussian function
        self.psi0 = self.gaussianPacket(self.x, self.x0, self.kappa, self.sigma, norm=True)
        
        # Calculate potential energy based on the chosen potential type
        self.V = self.getPotential(potential)

        # Calculate matrices A and B for time evolution
        self.A, self.B = self.getAB()

        # Initialize array to store wavefunction psi
        self.psi = []

    def getPotential(self, choice):
        # Method to define the potential energy based on the chosen potential type
        
        # Infinite square well potential
        if choice == "ISW":
            return self.infiniteSquareWell()
        
        # Tunneling potential
        elif choice == "tunneling":
            return self.tunneling(self.x, self.L/4, self.L/100, 10)
        
        # Invalid potential type
        else:
            print('Choose a potential between: (ISW, Tunneling)')

    def infiniteSquareWell(self):
        # Method to define the infinite square well potential
        
        pot = np.zeros(len(self.x))  # Initialize potential array with zeros
        pot[0] = 1e10  # Set potential to infinity at boundaries
        pot[-1] = 1e10
        return pot
    
    def tunneling(self, x, x1, width, amp):
        # Method to define the tunneling potential
        
        pot = np.zeros(len(x))  # Initialize potential array with zeros
        cond = (x >= x1) & (x <= x1 + width)  # Define condition for non-zero potential
        pot[cond] = amp  # Set potential to a specified amplitude within the defined region
        return pot
    
    def getAB(self):
        # Method to calculate matrices A and B for Crank Nicholson
        
        # Define coefficients alpha and beta
        alpha, beta = 1j * self.dt/(2 * self.dx ** 2), 1j * self.dt / 2
        
        # Initialize matrices A and B with zeros
        A = np.zeros((self.Nx, self.Nx), dtype=complex)
        B = np.zeros((self.Nx, self.Nx), dtype=complex)
        
        # Construct matrices A and B
        for i in range(self.Nx):
            A[i, i] = 1 + alpha + beta * self.V[i]
            B[i, i] = 1 - alpha - beta * self.V[i]
            if i > 0:
                A[i, i-1] = -alpha/2
                B[i, i-1] = alpha/2
            if i < self.Nx-1:
                A[i, i+1] = -alpha/2
                B[i, i+1] = alpha/2
        return A, B
    
    def gaussianPacket(self, x, x0, kappa, sigma, norm=True):
        # Method to generate a Gaussian wave packet
        
        a = 1 / (sigma * np.sqrt(2 * np.pi))  # Gaussian prefactor
        b = np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))  # Gaussian exponent
        c = np.exp(1j * kappa * (x-x0))  # Phase factor
        packet = a * b * c  # Complete wave packet
        
        # Normalize wave packet if required
        if norm: 
            normalization = np.sqrt(1 / np.trapz(np.abs(packet) ** 2, x=x))
            return normalization * packet
        else:
            return packet 
               
    def solve(self):
        # Method to solve the time-dependent Schrödinger equation.
        
        # Initialize array to store wavefunction psi
        self.psi = np.zeros((self.Nx, self.Nt), dtype=complex)
        self.psi[:, 0] = self.psi0  # Assign initial wavefunction at t=0
        
        # Calculate inverse of matrix A for time evolution
        invA = np.linalg.inv(self.A)
        
        # Time evolution loop
        for time in tqdm.tqdm(range(self.Nt - 1)):
            Y = self.B.dot(self.psi[:, time])  # Calculate Y vector for time step
            self.psi[:, time+1] = invA.dot(Y)  # Update wavefunction for next time step
        return self.psi
