# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:09:22 2015

@author: Austin Paul, Bryn Reinstadler
"""

"""
Current implementation takes as input:
Nuclear repulsion energy (enuc.dat)
Overlap integrals (s.dat)
KE integrals (t.dat)
nuclear attraction integrals (v.dat)
EE repulsion integrals (eri.dat)
"""

from urllib2 import urlopen
import numpy
import time

# This suppresses the scientific notation of the matrix representations
numpy.set_printoptions(suppress=True)

# Read in the necessary integral values from the web

baseurl = "http://sirius.chem.vt.edu/~crawdad/programming/project3/"
molec = {1: "h2o_sto3g", 2: "h2o_dz", 3: "h2o_dzp"}

print molec
choice = input("\nMolecule (1-3): ")
RunMP = input("\nDo you want to run this with MP2 correction? (1 for yes; 0 for no) \
              MP is not recommended for molecules 2 and 3: ")

if(RunMP == 0):
    RunMP = False
else:
    RunMP = True

# n is the number of electrons
n = 10

# Read in the electron-nuclear terms
enuc = numpy.loadtxt(urlopen(baseurl+molec[choice]+"/enuc.dat"))

# Read in the overlap, kinetic energy, and potential energy integrals
integrals = []  # 0 is s, 1 is t, 2 is v
for ext in ["/s.dat", "/t.dat", "/v.dat"]:
    f = urlopen(baseurl+molec[choice]+ext)
    i, j, val = numpy.loadtxt(f, unpack=True)
    integrals.append(zip(i, j, val))

# Read in the electron-electron repulsion integrals
f = urlopen(baseurl+molec[choice]+"/eri.dat")
p, q, r, s, val = numpy.loadtxt(f, unpack=True)
eris = zip(p, q, r, s, val)

"""
We get b from the last entry in the ERIs.
b is the number of bases we are using.
Then we make matrices of the various integrals.
"""

b = int(eris[-1][0])

ERI = numpy.zeros((b, b, b, b))
S = numpy.matrix(numpy.zeros((b, b)))
T = numpy.matrix(numpy.zeros((b, b)))
V = numpy.matrix(numpy.zeros((b, b)))

for row in eris:
    p = row[0]-1
    q = row[1]-1
    r = row[2]-1
    s = row[3]-1
    ERI[(p, q, r, s)] = ERI[(q, p, r, s)] = ERI[(p, q, s, r)] = ERI[(q, p, s, r)] = \
        ERI[(r, s, p, q)] = ERI[(r, s, q, p)] = ERI[(s, r, q, p)] = ERI[(s, r, p, q)] = row[-1]
for row in integrals[0]:
    i = row[0]-1
    j = row[1]-1
    S[(i, j)] = S[(j, i)] = row[-1]
for row in integrals[1]:
    i = row[0]-1
    j = row[1]-1
    T[(i, j)] = T[(j, i)] = row[-1]
for row in integrals[2]:
    i = row[0]-1
    j = row[1]-1
    V[(i, j)] = V[(j, i)] = row[-1]


# The core Hamiltonian is given by Huv = Tuv + Vuv
H = V+T

# Build an orthogonalization matrix
# BOOK STEP 3
# U = eigenvectors (columns)
# s = matrix with eigenvalues^-1/2 on diagonals
# Let's create an orthonormal basis
# Coefficients will be given by A = S^-1/2 = Us^(-1/2)U.H, and
# A.H*SA = I. The .H returns the conjugate transpose.

eigenvalues, U = numpy.linalg.eig(S)
s = numpy.diag([1/numpy.math.sqrt(x) for x in eigenvalues])

A = U*s*U.T

# We need to guess coefficients.
# Approximate H = F. Then epsilons are eigenvalues.

initFPrime = A.T*H*A
epsilons, Cprimes = numpy.linalg.eig(initFPrime)

# We must sort the epsilons (energies) so that we
# fill the lowest energy orbitals first in the
# density matrix.

ordering = epsilons.argsort()
SortCPrime = Cprimes[:, ordering]

C = A*SortCPrime

# Based on the equation after 14.57
# Now we calculate P, the density matrix, based on 14.42

P = numpy.zeros((b, b))

for t in range(b):
    for u in range(b):
        for j in range(n/2):
            P[t, u] += 2. * numpy.ma.conjugate(C[t, j])*C[u, j]

# Next, we calculate the initial energy
# So that we can use it to determine convergence
# We are using using formula (14.44) or (14.45).

elecEn = 0

newElecEn = 0
epsilons.sort()
for i in range(n/2):
    elecEn += epsilons[i]

for r in range(b):
    for s in range(b):
        elecEn += (0.5)*(P[r, s] * H[r, s])

"""
SCF Method

This part is the SCF method. Here we use F, the Fock operator,
which we guess to be H and then perturb using the density matrix
and the electron-electron repulsion terms.

We then continue the same process we did for the first time above
and compare the new electronic energy to the initial electronic energy
which was calculated above for convergence. We may change the convergence
limit if necessary, but currently it is set to 10e-12.
"""

converged = False
while(not converged):
    # Guess that the Fock operator is the Hamiltonian
    F = H.copy()
    for t in range(b):
        for u in range(b):
            F += P[t, u]*(ERI[:, :, t, u] - 0.5 * ERI[:, u, t, :])

    # Translate F into the ortho-normal basis
    Fprime = A.T*F*A
    epsilons, Cprimes = numpy.linalg.eig(Fprime)

    # Perform the same sorting as necessary earlier
    ordering = epsilons.argsort()
    SortCPrime = Cprimes[:, ordering]

    C = A*SortCPrime

    # We use the C's to calculate the new density matrix, P
    newP = numpy.zeros((b, b))

    for t in range(b):
        for u in range(b):
            for j in range(n/2):
                newP[t, u] += 2. * numpy.ma.conjugate(C[t, j]) * C[u, j]

    P = newP.copy()

    # Now we calculate the energy again after having self-consistently
    # changed the previous.
    newElecEn = 0
    epsilons.sort()
    for i in range(n/2):
        newElecEn += epsilons[i]

    for r in range(b):
        for s in range(b):
            newElecEn += (0.5)*(P[r, s] * H[r, s])

    # If the energies converge, exit the loop.
    if (numpy.abs(newElecEn - elecEn) < 10e-12):
        converged = True
    print newElecEn
    elecEn = newElecEn

# Here we print the final energy
print "\n The elec energy from HF: ", elecEn
elecEn += enuc
print "The final energy from HF: ", elecEn


"""
This ends the Hartree Fock method.

Steps to complete MP2:

# Find the MO coefficients and MO energies computed in the HF program from Project 3

# Transform two eris into the MO basis

# Compute MP2 energy using the summation given at the website.
"""
if(RunMP):
    # We have already found the MO coefficients and MO energies
    # computed in the HF program from Project 3
    # Therefore, we now transform the eris into the MO basis.
    MP = numpy.zeros((b, b, b, b))
    print "here goes nothing"
    start = time.time()
    for s in range(b):
        for t in range(s):
            for u in range(s):
                for v in range(b):
                    for w in range(b):
                        for x in range(b):
                            for y in range(b):
                                for z in range(b):
                                    MP[s, t, u, v] = MP[t, s, u, v] = MP[u, t, s, v] = \
                                        C[s, w]*C[t, x]*ERI[s, t, u, v]*C[u, y]*C[v, z]
    end = time.time()

    print "time: ", end-start
    # print "This is MP ", MP

    # Compute MP2 energy using the summation given at the website.
    EMP = 0.

    for i in range(n/2):
        for j in range(n/2):
            for a in range(n/2, b):
                for c in range(n/2, b):
                    EMP += (MP[i, a, j, c]*(2.*MP[i, a, j, c] - MP[i, c, j, a])) / \
                        (epsilons[i] + epsilons[j] - epsilons[a] - epsilons[c])

    print "\nEMP ", EMP

    print "Final energy, with MP2 correction: ", (elecEn + EMP)
