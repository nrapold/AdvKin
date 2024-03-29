# Skeleton for Exercise 2

# Imported modules
import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1

''' Exercise 1 classical and Wigner sampling functions '''
# INPUT:
# beta:      inverse temperature
# m:         mass of the oscillator
# w:         frequency of the oscillator
# OUTPUT:
# x,p:       array with Ns sampled values for x, p, respectively
def HO_rho_cl(beta, m, w):
    x_mean  = 0
    x_std = 1/(w*np.sqrt(beta*m))
    p_mean  = 0
    p_std = np.sqrt(m/beta)
    x = np.random.normal(x_mean,x_std)
    p = np.random.normal(p_mean,p_std)
    return (x,p)

def HO_rho_W(beta, m, w):
    alpha = np.tanh(beta*hbar*w/2)
    x_mean  = 0
    x_std = np.sqrt(hbar/(2*alpha*m*w))
    p_mean  = 0
    p_std = np.sqrt(m*w*hbar/(2*alpha))
    x = np.random.normal(x_mean,x_std)
    p = np.random.normal(p_mean,p_std)
    return (x,p)


''' Exercise 2 '''

''' 1.a) Calculate the classical trajectory of a system described by F(x) '''
# INPUT:
# force:  function that takes the position x as input and returns the force
# x0:     initial position
# p0:     initial momentum
# dt:     value of the time step
# Nt:     total number of steps
# m:      mass
# OUTPUT:
# x:    array of positions     (x(0),x(dt),x(2*dt),...,x(N*dt))
# p:    array of momenta       (p(0),p(dt),p(2*dt),...,p(N*dt))

# define velocity verlet algorythm
p_intermediate = lambda x0, p0, force, dt: p0+force(x0)*dt/2

def x_verlet(x0, p0, force, dt, m):
    return x0 + p_intermediate(x0,p0,force,dt)*dt/m
def p_verlet(x0, p0, force, dt, m):
    return p_intermediate(x0, p0, force, dt)+force(x_verlet(x0, p0, force, dt, m))*dt/2

#returns the trajectory and the momenta of a startposition x0, a startmomentum p0
#calculated with the verlet algorhythm with Nt timetsteps (size dt)
def trajectory_cl(x0, p0, force, dt, Nt, m):
    x = np.array([])
    p = np.array([])
    for i in range(0, Nt):
        x = np.append(x, x0)
        p = np.append(p, p0)
        x0 = x_verlet(x[-1], p[-1],force, dt, m)
        p0 = p_verlet(x[-1], p[-1],force, dt, m)

        ##first implementation. it doesnt work, although, i still think, that 
        ##this is what is written in teh exercise

        #x_temp = x_verlet(x0, p0,force, i*dt, m)
        #p_temp = p_verlet(x0, p0,force, i*dt, m)
        #x = np.append(x, x_temp)
        #p = np.append(p, p_temp)
    return (x,p)



''' 1.c) Simulate harmonic oscillation '''
def harmonic_force(x):
    return -m*x*w**2

def harmonic_energy(x,p):
    return (p**2)/(2*m) + 0.5*m*(w*x)**2


# ** fill in your model parameters **
m  = 0.1
w  = 2*np.pi
x0 = 1
p0 = 0
dt = 0.01
Nt = 500    

t = dt*np.arange(Nt)           # x axis values
(x,p) = trajectory_cl(x0, p0, harmonic_force, dt, Nt, m)    # (x(t),p(t))
E = harmonic_energy(x,p)         # energy E(t) ** implement your formula **



plt.figure()
plt.plot(t,x   , label = 'x(t)')
plt.plot(t,p   , label = 'p(t)')
plt.plot(t,E   , label = 'E(t)')
plt.title('Trajectory of harmonic oscillator')
plt.xlabel('Time')
plt.legend(loc = 'lower left')

''' 1.d) Sampling of initial states and correlation functions '''
Ns = 200        # increase this value after you debugged your code
beta = np.array([1, 100, 100000000, 1000000000000000000])

# Define whether you want classical or quantum statistics
#theory = 'Classical'
theory = 'Wigner'

# Correlation function: Define matrix where each line is the array
# representing <A(0) B(t)> for a given beta
# So the number of rows should be the number of betas and the number of
# columns is the number of times steps Nt
x0_x = np.empty((beta.size,Nt))
p0_p = np.empty((beta.size,Nt))
E    = np.empty((beta.size,Nt))

for i in range(beta.size):

    x0_x_i = np.zeros(Nt)    # This will be <x0 x(t)> for beta[i] later
    p0_p_i = np.zeros(Nt)    # This will be <p0 p(t)> for beta[i] later
    E_i    = np.zeros(Nt)    # This will be <E(t)> for beta[i] later

# NOTE: np.empty, unlike np.zeros, does not set the array values to zero, 
# and may therefore be marginally faster. On the other hand, it 
# requires the user to manually set all the values in the array, 
# and should be used with caution.

    # sample initial values and run trajectory
    for j in range(Ns):
        # sample initial values
        if theory == 'Classical':
            (x0,p0) = HO_rho_cl(beta[i], m, w) 
        else:
            (x0, p0) = HO_rho_W(beta[i], m, w)

        # x_ij is the trajectory for beta[i] and the j-th initial point
        # the same for p_ij
        (x_ij,p_ij) = trajectory_cl(x0, p0, harmonic_force, dt, Nt, m)
        x0_x_i += x_ij[0]*x_ij
        p0_p_i += p_ij[0]*p_ij
        E_i    += harmonic_energy(x_ij, p_ij)

    # Normalize sum to obtain average
    x0_x_i /= Ns
    p0_p_i /= Ns
    E_i    /= Ns

    # Assign to the corresponding matrix row
    x0_x[i,:] = x0_x_i
    p0_p[i,:] = p0_p_i
    E[i,:]    = E_i


plt.figure()
for i in range(beta.size):
    plt.plot(t,x0_x[i,:]     , label = r'$\beta = %0.1f$' % beta[i])

plt.title(theory + r' $\left\langle x_0 x(t) \right \rangle$')
plt.legend(loc = 'upper right')
plt.xlabel('Time')

plt.figure()
for i in range(beta.size):
    plt.plot(t,p0_p[i,:]     , label = r'$\beta = %0.1f$' % beta[i])

plt.title(theory + r' $\left\langle p_0 p(t) \right \rangle$')
plt.legend(loc = 'upper right')
plt.xlabel('Time')

plt.figure()
for i in range(beta.size):
    plt.plot(t,E[i,:]/(hbar*w)     , label = r'$\beta = %0.1f$' % beta[i])
plt.title(theory + r' $\left\langle E(t) \right \rangle$')
plt.legend(loc = (0.75,0.7))
plt.ylim((0,1.7))
plt.yticks(np.arange(9)*0.25)
plt.xlabel('Time')
plt.ylabel('$E/\hbar\omega$')
plt.yticks(np.arange(9)*0.25)

''' 1.e) Bonus figure '''
# Here goes your code (everything you need is already somewhere above)




plt.show()
