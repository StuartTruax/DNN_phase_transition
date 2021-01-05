import numpy as np
import matplotlib.pyplot as plt, matplotlib.cm as cm
import pandas as pd
import ipdb

###############
#
# basic methodology from:
# https://rajeshrinet.github.io/blog/2014/ising-model/#Monte-Carlo-simulation-of-2D-Ising-model
#
################

def generate_square_lattice_state(N_lattice):

    state = np.zeros((N_lattice,N_lattice))

    for i in range(N_lattice):
        for j in range(N_lattice):
            trial = np.random.binomial(1,0.5)
            if trial == 0:
                state[i,j]  = 1.0
            else:
                state[i,j]  = -1.0

    return state

def metropolis_increment(state, beta,J):

    N = state.shape[0]


    #perform  N^2 trials
    for index in range(N**2):
        i = np.random.randint(0,N)
        j = np.random.randint(0,N)

        spin_to_flip = state[i,j]

        neighbor_spin_sum = state[(i+1)%N,j]+state[i,(j+1)%N] +\
                            state[(i-1)%N,j]+state[i,(j-1)%N]

        # calculate the change in Hamiltonian from a flip in spin at the site.
        # this expression calculates the energy delta relative to E=0 and then
        # multiplies by 2 to find the change in energy due to flipping
        dE = 2*J*spin_to_flip*neighbor_spin_sum

        #if the flip lowers the energy, flip
        # otherwise accept move with thermal probability to
        # satisfy detailed balance (i.e. equilibrium) condition
        if dE < 0:
            spin_to_flip *=-1
        elif np.random.rand() < np.exp(-dE*beta):
            spin_to_flip *=-1

        state[i,j] = spin_to_flip

    return state


def calculate_energy(state,J,h=0):

    N = state.shape[0]

    H = 0

    for i in range(N):
        for j in range(N):
            #sum the coupling energy from all relevant neighbor pairings
            neighbor_spin_sum = state[(i+1)%N,j]+state[i,(j+1)%N] +\
                                state[(i-1)%N,j]+state[i,(j-1)%N]

            H+=-J*neighbor_spin_sum*state[i,j]

            H+=-h*state[i,j]

    H/=4.  #divide by 4 for overcounting

    return H


def calculate_magnetization(state):
    return np.sum(state)



if __name__=="__main__":

    np.random.seed(0)

    N_lattice=64
    N_temp_points = 256

    J = 1.0 #specify a ferromagnetic system
    k = 1.38*10**-23
    T_c  = 2*J/(k*np.log(1+np.sqrt(2)))

    temp_points = np.linspace(1.53, 3.28, N_temp_points) # kT
    equilibriation_steps = 1024 #important to have many steps
    calculation_steps = 1024 #important to have many steps

    energies = []
    magnetizations = []
    susceptibilities = []

    states = []


    #main monte-carlo loop
    #loop across the temperature range
    for t_index,temp_point in enumerate(temp_points):
        print("%d Temperature: %.3f"%(t_index,temp_point))

        energy = 0
        magnetization = 0
        magnetization_squared = 0

        state = generate_square_lattice_state(N_lattice)

        #equilibration moves
        for i in range(equilibriation_steps):
            metropolis_increment(state, J, 1.0/temp_point)

        #calculation moves
        for i in range(calculation_steps):
            metropolis_increment(state, J, 1.0/temp_point)

            delta_energy= calculate_energy(state,J)
            delta_magnetization = calculate_magnetization(state)

            energy+=delta_energy
            magnetization+=delta_magnetization
            magnetization_squared+= delta_magnetization**2



        #calculate pref-factors for averaging the thermodynamic variables
        prefactor_1 = 1.0/((calculation_steps)*(N_lattice**2))
        prefactor_2 = 1.0/((calculation_steps**2)*(N_lattice**2))

        #append variables
        susceptibility = (magnetization_squared*prefactor_1-\
                         magnetization*magnetization*prefactor_2)*temp_point

        energies.append(energy*prefactor_1)
        magnetizations.append(np.abs(magnetization*prefactor_1))
        susceptibilities.append(susceptibility)
        states.append(np.array(state))


    plt.figure(figsize=(5,5))
    plt.subplot(3,1,1)
    plt.plot(temp_points, energies, '.')
    plt.ylabel("Energy")
    plt.axvline(x=k*T_c, color='r')
    plt.subplot(3,1,2)
    plt.plot(temp_points, magnetizations, '.')
    plt.axvline(x=k*T_c, color='r')
    plt.ylabel("Magetization")
    plt.subplot(3,1,3)
    plt.plot(temp_points, susceptibilities, '.')
    plt.axvline(x=k*T_c, color='r')
    plt.ylabel("Susceptibility")
    plt.xlabel("Temperature")
    plt.show()


    N_states_to_plot = 4

    indices = np.linspace(0,len(states)-1,N_states_to_plot)

    plt.figure(figsize=(5,5))

    for i, _ in enumerate(indices):
        plt.subplot(N_states_to_plot,1,i+1)
        plt.imshow(states[int(indices[i])])
        plt.ylabel('T=%0.3f'%temp_points[int(indices[i])])

    plt.show()


    data = pd.DataFrame()

    data["lattice_state"] = states
    data["susceptibility"] = susceptibilities
    data["energy"] = energies
    data["magnetization"] = magnetizations
    data["temperature"] = temp_points
    data["above_T_c"] = np.sign(temp_points-k*T_c)

    try:
        data.to_pickle("states_%d_%d.pkl"%(N_lattice,N_temp_points))
    except:
        print("Failed to write file")
