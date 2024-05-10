# SchrodingerCrankNicholson

## 'solver.py'
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

## `utils.py`

This file contains functions used to visualize various aspects of the simulation results generated by the `Schrodinger` class defined in `solver.py`.

### Function: `animRealImag(schrodinger)`

Generates an animated plot showing the real and imaginary parts of the wave function as it evolves over time.

### Function: `animModulus(schrodinger)`

Generates an animated plot showing the modulus squared of the wave function as it evolves over time.

### Function: `plotUncertainty(schrodinger)`

Plots the uncertainty (variance) of the wave function's position over time.

### Function: `plotNormalization(schrodinger)`

Plots the normalization of the wave function over time.

### Function: `plotExpectedPosition(schrodinger, ret=False)`

Plots the expected position of the particle over time. If `ret=True`, returns the expected position array instead of plotting it.

### Function: `plotTimeEvolution(schrodinger)`

Plots the time evolution of the wave function as a color map, showing its magnitude at each position and time.

### Function: `plotSome(schrodinger)`

Plots real and imaginary parts of the wave function at various time steps.

