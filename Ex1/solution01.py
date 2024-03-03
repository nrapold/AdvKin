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

## (1a)
def sample_cl_x(beta, mass, omega, N):
    # Returns an array of positions, x, of length N,
    # sampled from the classical thermal distribution
    # for a harmonic oscillator.
    mu = 0.0
    sigma = 1.0/np.sqrt(beta*mass*omega**2)
    return np.random.normal(mu, sigma, N)

def sample_cl_p(beta, mass, omega, N):
    # Returns an array of momenta, p, of length N,
    # sampled from the classical thermal distribution
    # for a harmonic oscillator.
    mu = 0.0
    sigma = np.sqrt(mass/beta)
    return np.random.normal(mu, sigma, N)

## (1b)
def sample_qm_x(beta, mass, omega, hbar, N):
    # Returns an array of positions, x, of length N,
    # sampled from the Wigner thermal distribution
    # for a harmonic oscillator.
    mu = 0.0
    alpha = np.tanh(beta*hbar*omega/2)
    sigma = np.sqrt(hbar/(2*mass*omega*alpha))
    return np.random.normal(mu, sigma, N)

def sample_qm_p(beta, mass, omega, hbar, N):
    # Returns an array of momenta, p, of length N,
    # sampled from the Wigner thermal distribution
    # for a harmonic oscillator.
    mu = 0.0
    alpha = np.tanh(beta*hbar*omega/2)
    sigma = np.sqrt(mass*hbar*omega/(2*alpha))
    return np.random.normal(mu, sigma, N)

## (1c) and (1d)

def harmonic_potential(mass, omega, x):
    # Returns the potential energy at coordinate x in a
    # harmonic oscillator; the function also accepts arrays of x.
	return 0.5*mass*(omega*x)**2

def ground_state_density(mass, omega, hbar, x):
    # Returns the ground-state probability density at position x for
    # a quantum harmonic oscillator; the function also accepts
    # arrays of x.
    ans = np.exp(-(mass*omega*x**2/hbar))
    ans *= np.sqrt(mass*omega/(np.pi*hbar))
    return ans

"""
It is convenient to plot the results for parts (c) and (d)
as subplots on the same figure
"""

# create a figure comprised of two subplots, where subplot ax_c
# is for exercise (1c) and subplot ax_d is for exercise (1d);
fig, (ax_c, ax_d) = plt.subplots(ncols=2)
# get current figure size; the result is a list with
# figsize[0] giving the width and figsize[1] giving the height
figsize = fig.get_size_inches()
# Reduce the height
figsize[1] *= 0.55
fig.set_size_inches(figsize)

# Set the parameters
mass = 1.0
omega = 1.0
hbar = 1.0
N = 100000
nbins = 100

beta = 1.0
# Adjust x-axis plotting range
ax_c.set_xlim([-3,3])
# Label the axis. The "r" indicates that what follows is a raw
# string. You can read more about this online
ax_c.set_xlabel(r'x')
# Adjust y-axis plotting range and add label
ax_c.set_ylim([0,1.5])
ax_c.set_ylabel(r'ρ(x)')
# Set plot label
ax_c.set_title(r'β = 1')

# plot the potential
x = np.linspace(-3.0, 3.0, 251)
V = harmonic_potential(mass, omega, x)
ax_c.plot(x, V, label=r'V(x)')
# Classical statistics
x_cl_samples = sample_cl_x(beta, mass, omega, N)
ax_c.hist(x_cl_samples, density=True, alpha=0.5, bins=nbins,
          label=r'classical')
# Quantum statistics
x_qm_samples = sample_qm_x(beta, mass, omega, hbar, N)
ax_c.hist(x_qm_samples, density=True, alpha=0.5, bins=nbins,
          label=r'Wigner')
# Ground-state probability density
rho_ground = ground_state_density(mass, omega, hbar, x)
ax_c.plot(x, rho_ground, label=r'|ψ0(x)|²')
ax_c.legend(loc='upper right', ncol=1)

# Repeat for beta = 10
beta = 10.0
ax_d.set_xlim([-3,3])
ax_d.set_xlabel(r'x')
ax_d.set_ylim([0,1.5])
ax_d.set_ylabel(r'ρ(x)')
ax_d.set_title(r'β = 10')
ax_d.plot(x, V, label=r'V(x)')
# Classical statistics
x_cl_samples = sample_cl_x(beta, mass, omega, N)
ax_d.hist(x_cl_samples, density=True, alpha=0.5, bins=nbins,
          label=r'classical')
# Quantum statistics
x_qm_samples = sample_qm_x(beta, mass, omega, hbar, N)
ax_d.hist(x_qm_samples, density=True, alpha=0.5, bins=nbins,
          label=r'Wigner')
# Ground-state probability density
rho_ground = ground_state_density(mass, omega, hbar, x)
ax_d.plot(x, rho_ground, label=r'|ψ0(x)|²')
ax_d.legend(loc='upper right', ncol=1)

fig.tight_layout()
fig.savefig(save_directory + 'rho_x.png')
fig.savefig(save_directory + 'rho_x.pdf')


## (1f)
for beta in [1, 10]:
    print()
    print(r'β = {:g}'.format(beta))
    x_cl = sample_cl_x(beta, mass, omega, N)
    p_cl = sample_cl_p(beta, mass, omega, N)
    x_qm = sample_qm_x(beta, mass, omega, hbar, N)
    p_qm = sample_qm_p(beta, mass, omega, hbar, N)
    print(r'<x>(cl) = {:.4f},  <x>(qm) = {:.4f}'.format(
        np.mean(x_cl), np.mean(x_qm)))
    print(r'<x²>(cl) = {:.4f},  <x²>(qm) = {:.4f}'.format(
        np.mean(x_cl**2), np.mean(x_qm**2)))
    print(r'<xp>(cl) = {:.4f},  <xp>(qm) = {:.4f}'.format(
        np.mean(x_cl*p_cl), np.mean(x_qm*p_qm)))


## (1g)
fig2, axarr = plt.subplots(nrows=2, ncols=2)
figsize = fig2.get_size_inches()
figsize[0] *= 0.8
fig2.set_size_inches(figsize)

def plot_2d_rho(x_samples, p_samples, ax, nbins, title):
    ax.hist2d(x_samples, p_samples, bins=nbins,
              density=True, cmap=cm.Blues,
              range=[[-2,2],[-2,2]])
    ax.set_aspect(1.0)
    ax.set_xlim([-2.0, 2.0])
    ax.set_xlabel(r'x')
    ax.set_ylim([-2.0, 2.0])
    ax.set_ylabel(r'p')
    ax.set_title(title)

nbins = 100
N=1000000
beta = 1.0
plot_2d_rho(sample_cl_x(beta, mass, omega, N),
            sample_cl_p(beta, mass, omega, N),
            axarr[0,0], nbins,
            r'classical (β = {:g})'.format(beta))
plot_2d_rho(sample_qm_x(beta, mass, omega, hbar, N),
            sample_qm_p(beta, mass, omega, hbar, N),
            axarr[1,0], nbins,
            r'Wigner (β = {:g})'.format(beta))
beta = 10.0
plot_2d_rho(sample_cl_x(beta, mass, omega, N),
            sample_cl_p(beta, mass, omega, N),
            axarr[0,1], nbins,
            r'classical (β = {:g})'.format(beta))
plot_2d_rho(sample_qm_x(beta, mass, omega, hbar, N),
            sample_qm_p(beta, mass, omega, hbar, N),
            axarr[1,1], nbins,
            r'Wigner (β = {:g})'.format(beta))
fig2.tight_layout()
fig2.savefig(save_directory + 'rho_px.png')
fig2.savefig(save_directory + 'rho_px.pdf')