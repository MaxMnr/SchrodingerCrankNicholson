# SchrodingerCrankNicholson

This project contains the implementation of the `Schrodinger` class, which serves as a solver for the Time-Dependent Schrödinger's Equation (TDSE) using the Crank-Nicholson method. The class provides a flexible framework to simulate the behavior of a 1D-particle trapped in a box under various potential conditions.

### Class: `Schrodinger`

#### Initialization Parameters:
- `Nx`: Number of spatial grid points.
- `Nt`: Number of time steps.
- `kappa`, `sigma`: Parameters for the Gaussian wave packet.
- `L`: Length of the spatial domain.
- `x0`: Initial position of the wave packet.
- `potential`: Type of potential (default is Infinite Square Well).
- `tmax`: Maximum time for simulation.

#### Methods:
- `getPotential(choice)`: Method to define the potential energy based on the chosen potential type.
- `infiniteSquareWell()`: Method to define the infinite square well potential.
- `tunneling(x, x1, width, amp)`: Method to define the tunneling potential.
- `getAB()`: Method to calculate matrices A and B for Crank-Nicholson.
- `gaussianPacket(x, x0, kappa, sigma, norm=True)`: Method to generate a Gaussian wave packet.
- `solve()`: Method to solve the time-dependent Schrödinger equation.
