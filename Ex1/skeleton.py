#!/usr/bin/python python3
# -*- coding: utf-8 -*-


# ensure backwards compatibility with Python 2
from __future__ import print_function, division
# load useful mathematical functions
import numpy as np
# load plotting module
import matplotlib.pyplot as plt
# load colour maps for 2d histograms
from matplotlib import cm
# load visualisation settings from 'plot.mplstyle', which should
# be present in the same directory as this script
plt.style.use(['plot.mplstyle'])
# get possibility to interact with os
import os
# create new directory to safe plots
save_directory = 'plots/'
os.makedirs(save_directory, exist_ok=True)
"""
Below, replace the keyword 'pass' with
code that calculates and returns samples of positions
and momenta, as specified in the comments.
You may find the following helpful:
    * https://docs.python.org/3/tutorial/introduction.html#numbers
    * https://docs.python.org/3/tutorial/controlflow.html#defining-functions
    * https://numpy.org/doc/stable/reference/routines.math.html
    * https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
    * (optional) https://numpy.org/doc/stable/user/basics.broadcasting.html

Please keep your code tidy and well commented, explaining why you are doing
what you are doing wherever necessary. Other good practices you should try 
to stick to include:
    * use consistent indentation, in this case 4 spaces rather than a tab
    * name your variables intelligently; often a slightly longer, more 
      descriptive name rather than a single letter helps avoid confusion 

"""

## (1a)
def sample_cl_x(beta, mass, omega, N):
    # Returns an array of positions, x, of length N,
    # sampled from the classical thermal distribution
    # for a harmonic oscillator.
    mean = 0.0
    variance = 1.0/np.sqrt(beta*mass*omega**2)
    return np.random.normal(mean, variance, N)


def sample_cl_p(beta, mass, omega, N):
    # Returns an array of momenta, p, of length N,
    # sampled from the classical thermal distribution
    # for a harmonic oscillator.
    mean = 0.0
    variance = np.sqrt(mass/beta)
    return np.random.normal(mean, variance, N)

## (1b)
def sample_qm_x(beta, mass, omega, hbar, N):
    # Returns an array of positions, x, of length N,
    # sampled from the Wigner thermal distribution
    # for a harmonic oscillator.
    alpha = np.tanh(beta*hbar*omega/2)
    mean = 0
    variance = np.sqrt(hbar/(2*alpha*mass*omega))
    return np.random.normal(mean, variance, N)

def sample_qm_p(beta, mass, omega, hbar, N):
    # Returns an array of momenta, p, of length N,
    # sampled from the Wigner thermal distribution
    # for a harmonic oscillator.
    alpha = np.tanh(beta*hbar*omega/2)
    mean = 0
    variance = np.sqrt((mass*omega*hbar*0.5)/alpha)
    return np.random.normal(mean, variance, N)

## (1c) and (1d)

"""
Replace 'pass' with code that returns the harmonic potential
energy and the ground-state density respectively
"""

def harmonic_potential(mass, omega, x):
    # Returns the potential energy at coordinate x in a
    # harmonic oscillator; the function also accepts arrays of x.
	return 0.5*mass*omega**2*x**2

def ground_state_density(mass, omega, hbar, x):
    # Returns the ground-state probability density at position x for
    # a quantum harmonic oscillator; the function also accepts
    # arrays of x.
    return np.sqrt((mass*omega)/np.pi*hbar)*np.exp((-mass*omega*x**2)/hbar)

"""
It is convenient to plot the results for parts (c) and (d)
as subplots on the same figure. You may find the following
helpful:
    * https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

"""

# create a figure made out of two subplots, where subplot ax_c is for
# exercise (1c) and subplot ax_d is for exercise (1d);
fig, (ax_c, ax_d) = plt.subplots(ncols=2)
# get current figure size; the result is a list with
# figsize[0] giving the width and figsize[1] giving the height
figsize = fig.get_size_inches()
# Reduce the height. The symbol '*=' denotes in-place
# multiplication and is equivalent in this case to
# writing figsize[1] = figsize[1] * 0.55
figsize[1] *= 0.55  # feel free to change this number and see how it affects the final figure
fig.set_size_inches(figsize)

# Set the parameters
mass = 1.0
omega = 1.0
hbar = 1.0
# Feel free to vary N and nbins, and see how it affects your plot
N = 1000
nbins = 100

"""
First plot the distributions for beta = 1.

Labelling the x-axis is done for you; the y-axis you
have to process yourself
"""
beta = 1.0

# Adjust x-axis plotting range
ax_c.set_xlim([-3,3])
# Label the axis. The "r" indicates that what follows is a raw
# string. You can read more about this on https://docs.python.org/3/tutorial/introduction.html#strings
ax_c.set_xlabel(r'x')
# Adjust y-axis plotting range and add label
ax_c.set_ylim([0, 1.5])
ax_c.set_ylabel(r'rho(x)')


# Set plot label
ax_c.set_title(r'β = 1')

# Plot the potential
x = np.linspace(-3.0, 3.0, 251)
V = harmonic_potential(mass, omega, x)
ax_c.plot(x, V, label=r'V(x)')

# Histogram the classical distribution - see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html for further info
x_cl_samples = sample_cl_x(beta, mass, omega, N)
ax_c.hist(x_cl_samples, density=True, alpha=0.5, bins=nbins,
          label=r'classical')
# Histogram the Wigner distribution - use the above example for
# classical data to write your own code

x_qm_samples = sample_qm_x(beta, mass, omega, hbar, N)
ax_c.hist(x_qm_samples, density=True, alpha=0.5, bins=nbins,
          label=r'quantum')

# Ground-state probability density - use the example plotting the
# potential energy to write your own code

x = np.linspace(-3.0, 3.0, 251)
V = ground_state_density(mass, omega, hbar, x)
ax_c.plot(x, V, label=r'Ground-state probdist')

# Add legend - see https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html#matplotlib.axes.Axes.legend
ax_c.legend(loc='upper right')


"""
Repeat the above steps for beta = 10, plotting all the data
on ax_d.
"""

beta = 10.0

# Adjust x-axis plotting range
ax_d.set_xlim([-3,3])
# Label the axis. The "r" indicates that what follows is a raw
# string. You can read more about this on https://docs.python.org/3/tutorial/introduction.html#strings
ax_d.set_xlabel(r'x')
# Adjust y-axis plotting range and add label
ax_d.set_ylim([0,1.5])
ax_d.set_ylabel(r'rho(x)')


# Set plot label
ax_d.set_title(r'β = 10')

# Plot the potential
x = np.linspace(-3.0, 3.0, 251)
V = harmonic_potential(mass, omega, x)
ax_d.plot(x, V, label=r'V(x)')

# Histogram the classical distribution - see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html for further info
x_cl_samples = sample_cl_x(beta, mass, omega, N)
ax_d.hist(x_cl_samples, density=True, alpha=0.5, bins=nbins,
          label=r'classical')
# Histogram the Wigner distribution - use the above example for
# classical data to write your own code

x_qm_samples = sample_qm_x(beta, mass, omega, hbar, N)
ax_d.hist(x_qm_samples, density=True, alpha=0.5, bins=nbins,
          label=r'quantum')

# Ground-state probability density - use the example plotting the
# potential energy to write your own code

x = np.linspace(-3.0, 3.0, 251)
V = ground_state_density(mass, omega, hbar, x)
ax_d.plot(x, V, label=r'Ground-state probdist')

# Add legend - see https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html#matplotlib.axes.Axes.legend
ax_d.legend(loc='upper right')

# The following automatically adjusts the spacing between subplots.
# See what happens if you comment this line out
fig.tight_layout()
# This saves the plot in PNG and PDF formats
fig.savefig(save_directory + 'my_rho_x.png')
fig.savefig(save_directory + 'my_rho_x.pdf')


## (1f)
"""
Because we are repeating the same steps for both temperatures,
it makes sense to put them inside a loop: see https://docs.python.org/3/tutorial/controlflow.html#for-statements
"""
for beta in [1, 10]:
    print()
    # The following is a formatted string. Read more about it
    # on https://docs.python.org/3/library/string.html#format-string-syntax
    print(r'β = {:g}'.format(beta))
    # Below create arrays x_cl, p_cl, x_qm, and p_qm, each of length
    # N, that contain positions (x) and momenta (p) drawn from
    # the classical (cl) and Wigner (qm) distributions respectively.

    x_cl = sample_cl_x(beta, mass, omega, N)
    p_cl = sample_cl_p(beta, mass, omega, N)
    x_qm = sample_qm_x(beta, mass, omega, hbar, N)
    p_qm = sample_qm_p(beta, mass, omega, hbar, N)

    # This prints the average x.
    print(r'<x>(cl) = {:.4f},  <x>(qm) = {:.4f}'.format(
        np.mean(x_cl), np.mean(x_qm)))

    # Based on the code above, print the average values of x² and px

    print(r'<x²>(cl) = {:.4f}, <x²>(qm) = {:.4f}'.format(
         np.mean(x_cl**2), np.mean(x_qm**2)))

    print(r'<px>(cl) = {:.4f}, <px>(qm) = {:.4f}'.format(
         np.mean(p_cl*x_cl), np.mean(p_qm*x_qm)))
## (1g)
fig2, axarr = plt.subplots(nrows=2, ncols=2)
"""
Here the figure is made of (2x2) subplots, positioned on a grid like

+---------------+---------------+
|               |               |
|               |               |
|  axarr[0,0]   |  axarr[0,1]   |
|               |               |
|               |               |
+---------------+---------------+
|               |               |
|               |               |
|  axarr[1,0]   |  axarr[1,1]   |
|               |               |
|               |               |
+---------------+---------------+


"""

# Adjust the figure size as appropriate - use the example from before
# and make sure to replace fig with fig2

fig2size = fig2.get_size_inches()
fig2size[1] *= 1
fig2.set_size_inches(fig2size)

"""
Again, when making plots for part (g), we are repeating
essentially the same operation multiple times. To make our
lives easier we can put all the repeated steps inside a function:
"""
def plot_2d_rho(x_samples, p_samples, ax, nbins, title):
    """
    On the axis 'ax' plot a 2D histogram of the data in
    'x_samples' and 'p_samples', using a total of 'nbins'
    histogram bins along each axis. The axis is given the
    label 'title'.

    """

    # See https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist2d.html?highlight=hist2#matplotlib.axes.Axes.hist2d
    # for more details
    # You may change the colour map (cm.Blues) however you please
    # - see https://matplotlib.org/stable/gallery/color/colormap_reference.html for what options are available.
    ax.hist2d(x_samples, p_samples, bins=nbins,
              density=True, cmap=cm.Blues,
              range=[[-2,2],[-2,2]])
    # Set the aspect ratio of the subplot
    ax.set_aspect(1.0)
    # Below, set the ranges and labels of the x- and y-axes,
    # and give the subplot a title, which is passed to
    # this function as the variable 'title'
    ax.set_xlim([-2,2])
    ax.set_xlabel(r'x')
    ax.set_ylim([-2,2])
    ax.set_ylabel(r'p')
    ax.set_title(title)

# Feel free to change the number of bins and samples below
nbins = 50
N = 100000

# Plot the data for beta = 1.0 in the left column of the subplot grid.
beta = 1.0
# Classical data
plot_2d_rho(sample_cl_x(beta, mass, omega, N),
            sample_cl_p(beta, mass, omega, N),
            axarr[0,0], nbins,
            r'classical (β = {:g})'.format(beta))

# Quantum data - plot on axarr[1,0]
plot_2d_rho(sample_qm_x(beta, mass, omega, hbar, N),
            sample_qm_p(beta, mass, omega, hbar, N), 
            axarr[1, 0], nbins, r'quantum (β = {:g})'.format(beta))

# Repeat for beta = 10, now plotting data in the right column of
# the subplot grid. At the end, adjust the subplot spacing in 'fig2' as
# we have previously done fo 'fig', and save the resulting plot as
# PNG and PDF.

beta = 10.0
plot_2d_rho(sample_cl_x(beta, mass, omega, N),
            sample_cl_p(beta, mass, omega, N),
            axarr[0,1], nbins,
            r'classical (β = {:g})'.format(beta))

# Quantum data - plot on axarr[1,0]
plot_2d_rho(sample_qm_x(beta, mass, omega, hbar, N),
            sample_qm_p(beta, mass, omega, hbar, N), 
            axarr[1, 1], nbins, r'quantum (β = {:g})'.format(beta))

fig2.tight_layout()
fig2.savefig(save_directory + 'my_rho_px.png')
fig2.savefig(save_directory + 'my_rho_px.pdf')
