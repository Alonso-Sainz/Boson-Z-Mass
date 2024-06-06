# -*- coding: utf-8 -*-
"""
---------------Title---------------
PHYS20161 Final Assignment: Z Boson
-----------------------------------

It first reads in data from two different files and combines them. Before combining
it validates the data by:

1) Removing Nan in the file
2) Removing negative values
3) Removes data more than 3 times the mean
4) And sorts them in terms of energy

It then proceeds to do a minimised chi squared fit to the data by varying two parameters
simultaneously. With this fit it compares it to the data and removes the noise.
It then produces another minimised chi squared fit and returns the fitted paramaters
Finally it calculates the reduced chi squared and the lifetime of the boson.

@author: Student ID: 10458254 06/12/2021
"""

# IMPORT STATEMENTS

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


#  CONSTANTS

FILE_NAME_1 = 'z_boson_data_1.csv'
FILE_NAME_2 = 'z_boson_data_2.csv'
PARTIAL_WIDTH = 83.91
UNITS_CONVERSION = (197.33**2 / 1E6) * 10
H_BAR = 6.5821E-16


# FUNCTIONS

def read_data ():
    """
    Reads and joins two data files in to one

    Returns
    -------
    An array containing the both files: joint_file

    """
    try:
        input_file_1 = open(FILE_NAME_1, 'r')
        input_file_2 = open(FILE_NAME_2, 'r')
    except FileNotFoundError:
        print("The file cannot be found")

    file_1 = np.genfromtxt(input_file_1, delimiter= ',', comments= '%')
    file_2 = np.genfromtxt(input_file_2, delimiter= ',', comments= '%')



    joint_file = np.vstack((file_1, file_2))



    return joint_file

def validate_data (file_with_nan):
    """
    It validates the data by:
    1) Removing Nan in the file
    2) Removing negative values
    3) Removes data more than 3 times the avarege
    4) And sorts them in terms of energy

    Parameters
    ----------
    file_with_Nan : Array
        The array with the two data files combined

    Returns
    -------
    file_sorted : Array
        An array with all the data validated and sorted

    """
    # Remove Nan from data
    file_with_nan = (file_with_nan[~np.isnan(file_with_nan).any(axis=1)])

   # Remove Negative values

    file_with_errors = np.zeros((0,3))

    for line in file_with_nan:
        if line[0] > 0 and line[1] > 0 and line[2] > 0:

            file_with_errors = np.vstack((file_with_errors, line))

    # Remove noise from data

    file_unsorted =np.zeros((0,3))


    for line in file_with_errors:

        if line[1] < (3 * np.average(file_with_errors) ):

            file_unsorted = np.vstack((file_unsorted, line))


    # Sorts the file
    file_sorted = file_unsorted[np.argsort(file_unsorted[:, 0])]

    return file_sorted

def remove_noise(file_noise, fit_mz, fit_gammaz):
    """
    Removes data points that deviate more than 3 standard deviations from the fit curve

    Parameters
    ----------
    file_noise : Array
        An array with the values with noise in them
    fit_mz : Float
        The value of mz obtained for the fit
    fit_gammaz : Float
        The value of mz obtained for the fit

    Returns
    -------
    filterd_file : array
        returns the same file tha was inputed but with out noise

    """
    filterd_file = np.zeros((0,3))

    for line in file_noise:
        if np.abs((line[1] - cross_section(line[0], fit_mz, fit_gammaz))) < 3 * line[2]:

            filterd_file = np.vstack((filterd_file, line))

    return filterd_file

def cross_section(energy, boson_mass, boson_width):
    """
    Calculates the cross section of the Z boson collisons from elctron-positron collisons
    given by the following equation:
        σ = (12π * E^2 * Γee^2) / (m^2 * (E^2 − m^2)^2 + m^2 * Γz^2)

    Where m is the boson mass, Γee the partial width, Γz boson width.

    Parameters
    ----------
    energy : Float
    boson_mass : Float
    boson_width : Float

    Returns
    -------
    The cross section in mili-barns (Float)

    """


    return (((12*np.pi / boson_mass**2) * (energy**2 * PARTIAL_WIDTH**2)/
            ((energy**2 - boson_mass**2)**2 + boson_mass**2 * boson_width**2))
            * UNITS_CONVERSION)

def boson_lifetime (boson_width):
    """
    Calculates the boson lifetime given by equation:

        τ = h_bar / Γ

    Where Γ is the width

    Parameters
    ----------
    boson_width : Float
        From the fitted parameters

    Returns
    -------
    Lifetime of the boson in seconds (Float)

    """

    return H_BAR / (boson_width * 1E9) #Multiply by 10^9 to get GeV to eV

def chi_squared(data, boson_mass, boson_width):
    """


    Parameters
    ----------
    boson_mass : TYPE
        DESCRIPTION.
    boson_width : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    energy = np.array(data[:,0])
    cross = np.array(data[:,1])
    uncertainties = np.array(data[:,2])

    return np.sum(((cross_section(energy, boson_mass, boson_width) - cross)
                  / uncertainties)**2)

def reduced_chi_squared(data, boson_mass, boson_width):
    """
    Calculates the chi squared with the new fitted parameters and then calculates the
    reduced chi squared.

    Parameters
    ----------
    boson_mass : Float
        From the fitted parameters
    boson_width : Float
        From the fitted parameters

    Returns
    -------
    chi_squared : Float

    """

    return chi_squared(data, boson_mass, boson_width) / (len(data[:,0]) - 2)

def fitting_function (data):
    """
    Uses the curve fit fucntion from SciPy which finds the parameters with the least
    chi squared value. Which are the boson mass and width.

    Returns
    -------
    popt : An array with 2 elemnts [Float, Float]
        The fitted parameters boson mass and width in that order
    pcov : A 2 by 2 covariance array
        DESCRIPTION.

    """
    guesses = [np.average(data[:,0]),3]


    popt, pcov = curve_fit(cross_section, data[:,0], data[:,1], p0=(guesses),
                          sigma=(data[:,2]), absolute_sigma = True)
    # Calculates the uncertainty from covariance matrix
    perr = np.sqrt(np.diag(pcov))
    uncertainty_mz, uncertainty_gammaz = perr

    return popt, uncertainty_mz, uncertainty_gammaz

def plotting_function(data, fitting_data):
    """
    Plots a scattered graph of the data points and then plots the fitted line and saves the
    file as "ZBoson_plot.png". All in the same plot

    Parameters
    ----------
    data_1 : array of floats
        The file containing all the data.

    Returns
    -------
    None.

    """
    # plots the data

    x_points = np.array([])
    y_points = np.array([])

    x_points = data[:,0]
    y_points = data[:,1]

    plt.title("Z Boson")
    plt.xlabel("Energy (GeV)")
    plt.ylabel("Cross section (nb)")
    # plt.scatter(x_points, y_points, s=8)
    plt.errorbar(x_points, y_points, yerr =  data[:,2], color = "black", ecolor = "blue",
                 fmt="o", markersize = 2, elinewidth = 1)

    # plots the fit curve

    x_fit = data[:,0]
    y_fit = cross_section(x_fit, fitting_data[0], fitting_data[1])

    plt.plot(x_fit, y_fit, color="red")
    plt.legend(["Fit line", "Data points"])

    plt.savefig("ZBoson_plot.png", dpi = 600)
    plt.show()

    return 0

def contour_plot (data, boson_mass, boson_width):
    """
    Makes a contpur plot of chi squared when varying the boson mass and width found for
    the fit.

    Parameters
    ----------
    data : Array
        The data file with out noise
    boson_mass : Float
        From the fitted parameters
    boson_width : Float
        From the fitted parameters

    Returns
    -------
    None

    """
    # Makes 2 mesh arrays with the boson mass and width
    mz_values = np.linspace(boson_mass - 0.05, boson_mass + 0.05,100)
    gammaz_values = np.linspace(boson_width - 0.05, boson_width + 0.05 ,100)

    x_values_mesh = np.empty((0,100)) #boson mass

    y_values_mesh = np.empty((0,100)) #boson width
    chi_squared_mesh = np.empty((0,100))

    for _ in mz_values:
        x_values_mesh = np.vstack((x_values_mesh, mz_values))

    for _ in gammaz_values:
        y_values_mesh = np.vstack((y_values_mesh, gammaz_values))
    y_values_mesh = np.transpose(y_values_mesh)


    # Produces a chi_squared_mesh
    for rows in range(len(x_values_mesh)):

        dummy_array = np.array([])

        for columns in range(len(x_values_mesh)):

            dummy_array = np.append(dummy_array, chi_squared(data, x_values_mesh[rows,columns],
                                           y_values_mesh[rows,columns]))

        chi_squared_mesh = np.vstack((chi_squared_mesh,dummy_array))

    # Produces the contour plot
    chi_squared_levels = (np.amin(chi_squared_mesh) + 2.30,
                          np.amin(chi_squared_mesh) + 5.99,
                          np.amin(chi_squared_mesh) + 9.21)

    contour = plt.contour(x_values_mesh, y_values_mesh, chi_squared_mesh,
                          levels = chi_squared_levels)
    plt.scatter(boson_mass, boson_width, marker=("X"), s = 50)

    plt.clabel(contour, fontsize = 10)
    plt.title("Chi squared")
    plt.xlabel("Boson mass (GeV/c^2)")
    plt.ylabel("Boson width (GeV)")
    plt.savefig("ZBoson_contour.png", dpi = 600)

    return 0

def main():
    """
    It reads and combines both data files and validates them. It attempts to
    make a fit line and removes the noise. Then it finds the best fit line and plots
    it and the data as well as a contour plot of chi squared.

    Returns
    -------
    None.

    """
    # Read and combines both data files as well as validating the data
    raw_data = read_data()
    data = validate_data(raw_data)

    # Produces a first attempt to make a fit
    fitting_parameters, mz_uncertainty, gammaz_uncertainty = fitting_function(data)

    # Removes noise based on the fit line
    data = remove_noise(data,fitting_parameters[0] , fitting_parameters[1])

    # Does a fit to the data with out noise
    fitting_parameters, mz_uncertainty, gammaz_uncertainty = fitting_function(data)

    #Plots the data and the fit line as well as a contour plot. Saves both files
    plotting_function(data,fitting_parameters)
    contour_plot(data, fitting_parameters[0], fitting_parameters[1])

    print("The mass of the boson is {0:.4f} ± {1:.4f} GeV/c^2".format(fitting_parameters[0],
                                                                      mz_uncertainty,))

    print("The width of the boson is {0:.4f}  ± {1:.4f} GeV".format(fitting_parameters[1],
                                                                    gammaz_uncertainty))

    print("The reduced chi-square is {0:.3f}".format(reduced_chi_squared(data,
                                            fitting_parameters[0], fitting_parameters[1])))

    print("The lifetime is {:.3e} s".format(boson_lifetime(fitting_parameters[1])))

    return 0

if __name__ == '__main__':
    main()
