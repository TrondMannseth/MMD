__author__ = 'svenn'

# External imports
import numpy as np
from scipy import linalg  # Linear algebra tools (from scipy rather than numpy; see scipy website)
import os, sys, glob
from datetime import datetime
from subprocess import call, DEVNULL

class Cholesky:
#class Geostat:
    """
    Class with various geo-statistical algorithms, s.a., generation of covariance, unconditional random variable, etc.
    """
    # TODO: Make it more like a class, and less like some functions that have been thrown in here for convenience.
    # TODO: Implementation of 3D fields (preferably with option for cross-covariance terms between the layers)

    def __init__(self):

        # Within
        self.real = None
        self.cov = None
        self.mean = None
        self.sgsim_input = None
        self.outfile = None
        self.number = None

    def gen_real(self, mean, var, number, limits=None, return_chol=False):
        """
        Function for generating unconditional random realizations of a variable using Cholesky decomposition.

        Input:
                - mean:         Mean vector or scalar
                - var:          (Co)variance
                - number:       No. of realizations

        Optional input:
                -limits:        Truncation limits
                -return_chol:   Boolean that indicaties if the sqrt of the covariance should be returned

        ST 18/6-15: Wholesale copy of code written by Kristian Fossum. Some modification has been done
        KF 15/6-16: Added option to return sqrt of matrix.
        """
        parsize = len(mean)
        if parsize == 1:
            l = np.sqrt(var)
            # tmp = mean + L*np.random.randn(1, number)
        else:
            # Cholesky decomposition
            l = linalg.cholesky(var)

        # Gen. realizations
        tmp = np.tile(np.reshape(mean, (len(mean), 1)), (1, number)) + np.dot(l.T, np.random.randn(np.size(mean),
                                                                                                   number))

        # Truncate values that are outside limits
        # TODO: Make better truncation rules, or switch truncation on/off
        if limits is not None:
            # Truncate
            tmp[tmp > limits['upper']] = limits['upper']
            tmp[tmp < limits['lower']] = limits['lower']

        self.real = tmp

        if return_chol:
            return self.real, l
        else:
            return self.real

    def gen_sGsim_real(self):
        """
        Function for running the GSLIB package sGsim. It is assumed that we already have run init_sgsim such that the
        input file is generated. It is further assumed that GSLIB is installed an in the system path. For more
        information regarding GSLIB, source code and executables see: http://www.gslib.com/
        """

        # Run sGsim
        call(['sgsim', self.sgsim_input], stdout=DEVNULL)

        with open(self.outfile, 'r') as file:
            lines = file.readlines()

        top_head = lines[0]
        info = lines[1].strip().split()
        head = lines[2].strip()

        assert head == 'value' # Head should be value or something might be wrong

        tmp = np.array([float(elem.strip()) for elem in lines[3:]]).reshape((self.number,
                                                                             float(info[1])*float(info[2]))).T
        # Must be transposed to match the general structure of the parameters
        if self.mean is not None:
            tmp += np.tile(self.mean.reshape(len(self.mean), 1), (1, self.number))


        # Remove all tmp files that sgsim creates
        for fl in glob.glob('sgsim*'):
            os.remove(fl)

        self.real = tmp
        return self.real

    def gen_sIsim_real(self):
        """
        Function for running the GSLIB package sIsim. It is assumed that we already have run init_sisim such that the
        input file is generated. It is further assumed that GSLIB is installed an in the system path. For more
        information regarding GSLIB, source code and executables see: http://www.gslib.com/
        """
        # Run sIsim
        call(['sisim', self.sisim_input], stdout=DEVNULL)

        with open(self.outfile, 'r') as file:
            lines = file.readlines()

        top_head = lines[0]
        info = lines[1].strip().split()
        head = lines[2].strip()

        assert head == 'Simulated Value' # Head should be value or something might be wrong

        tmp = np.array([float(elem.strip()) for elem in lines[3:]]).reshape((self.number,
                                                                             float(info[1])*float(info[2]))).T
        # Must be transposed to match the general structure of the parameters
        if self.mean is not None:
            # For the categorical values we multiply the mean and to the realizations
            for i in range(self.number):
                tmp[:, i] = self.mean*tmp[:, i]
        if self.facies_var is not None:
            for i in range(self.number):
                tmp[:, i] += self.facies_var[:, i]


        # Remove all tmp files that sgsim creates
        for fl in glob.glob('sisim*'):
            os.remove(fl)

        self.real = tmp
        return self.real


    def genCov(self, x_size, y_size, variance, var_range, aspect, angle, var_type):
        """
        Function for generating a stationary covariance matrix based on variogram models.

        Input:
                - x_size,y_size:        No. of grid cells in x and y direction
                - variance:             Sill
                - var_range:            Variogram range
                - aspect:               Ratio between x-axis (major axis) and y-axis
                - angle:                Rotation of the x-axis. Measured in degrees clockwise
                - var_type:             Variogram model

        Output:
                - cov:                  Covariance matrix (size: x_size x y_size)

        ST 18/6-15: Wholesale copy of code written by Kristian Fossum. Some modifications have been made...
        ------------------------------------------------------------------------------------------------
        KF 04/11-15: Added two new variogram models: exponentioal and cubic. Also updated the
                     coefficients in the spherical model.
        TM 07/09-22: Added Gaussian variogram model
        """
        # TODO: General input coordinates

        [xx, yy] = np.mgrid[1:x_size+1, 1:y_size+1]
        pos = np.zeros((xx.size, 2))
        #TM030621#pos[:, 0] = np.reshape(xx, xx.size, 1)
        #TM030621#pos[:, 1] = np.reshape(yy, yy.size, 1)
        pos[:, 0] = np.reshape(xx, xx.size)
        pos[:, 1] = np.reshape(yy, yy.size)

        d = np.zeros((xx.size, yy.size))

        for i in range(0, xx.size):
            jj = np.arange(0, yy.size)

            p1 = np.tile(pos[i, :], (yy.size, 1))
            p2 = pos[jj, :]

            dd = self._edist(p1, p2, aspect, angle)
            d[i, :] = dd

        # TODO: Variogram models in separate methods

        # Variogram models are for 1-d fields given by equations on pg. 641 in "Geostatistics Modeling spatial
        # uncertainty, J.P. Chiles and P. Delfiner, 2. ed, 2012
        if var_type == 'sph':
            s1 = np.nonzero(d < var_range)
            s2 = np.nonzero(d >= var_range)
            gamma = d*0
            gamma[s1] = variance - variance*((3/2)*np.fabs(d[s1])/var_range - (1/2)*(d[s1]/var_range)**3)
            gamma[s2] = 0

        if var_type == 'exp':
            gamma = variance*(np.exp(-np.fabs(3*d)/var_range))

        if var_type == 'gau':
            gamma = variance*(np.exp(-3*np.square(d/var_range)))

        if var_type == 'cub':
            s1 = np.nonzero(d < var_range)
            s2 = np.nonzero(d >= var_range)
            gamma = d*0
            gamma[s1] = variance*(1 - 7*(np.fabs(d[s1])/var_range)**2 + (35/4)*(np.fabs(d[s1])/var_range)**3 -
                                  (7/2)*(np.fabs(d[s1])/var_range)**5 + (3/4)*(np.fabs(d[s1])/var_range)**7)
            gamma[s2] = 0

        self.cov = gamma

        return self.cov

    def _edist(self, v1, v2, aspect, rotate):
        """
        Function for calculating the Euclidean distance of, possibly, anisotropic (rotated and scaled) vectors

        Input:
                - v1,v2:        Vectors to calculate distance between
                - r:            Range of variogram
                - aspect:       Ratio between the x-axis (major axis) and y-axis
                - rotate:       Rotation of the x-axis. Measured in degrees clockwise

        Output:
                - dist:         Euclidean distance between the v1 and v2.

        ST 18/6-15: Wholesale copy of code written by Kristian Fossum. Some modifications have been made...
        """
        # Rotation matrix
        RotMat = np.matrix([[np.cos((rotate/180)*np.pi), -np.sin((rotate/180)*np.pi)],
                            [np.sin((rotate/180)*np.pi), np.cos((rotate/180)*np.pi)]])

        # Stretching matrix
        RescaleMat = np.matrix([[1, 0], [0, aspect]])

        # Coordinates
        dp = v1 - v2

        # Do rotation and scaling
        dp = np.dot(RescaleMat*RotMat, dp.T)

        # Taken from org. GeoStat code:
        # For some reason, stretching is moved from y-coord to x-coord.
        dp = dp/aspect

        # Calc. distance
        dist = np.array(np.sqrt(np.sum(np.multiply(dp, dp), 0)))

        return dist

    def init_sgsim(self, x_size, y_size, data='foo.dat', var=1, mean=None, var_type='sph', outfile='sgsim.out',
                   corr_range=1, corr_aniso=1, corr_angl=0, limits=[-1.0e21, 1.0e21], number=1):
        """
        This script writes the input file for gslib's sequential Gaussian simulation package.
        Input:
                        - x_size: Size of field in the x direction
                        - y_size: Size of field in the y direction

        Optional inputs
                        - data: directory giving hard data constraints (must be in Geo-EAS format).
                                Default value does not exist
                        - var: variance value. (Sill)
                                Default value gives variance = 1
                        - mean: Stationary mean.
                                Default values gives 0
                        - var_type: Which variogram type should be selected. 1:Spherical, 2:exponential, 3:Gaussian
                                Default values gives a spherical variogram model
                        - outfile: Directory of the outfile where the field is written.
                                Default value if 'foo.out'.
                        - corr_range: Correlation range,
                                Default value = 1
                        - corr_aniso: Correlation anisotropy coefficient [0 1]
                                Default value = 1 (isotropic)
                        - corr_angl: Correlation angle (from y-axis)
                                Default value = 0 (correlation along the y-axis)
                        - limits: min and max truncation limits
                                Default values, min: -1.0e21, max: 1.0e21
                        - number: number of ensemble members
                                Default value = 1
        KF 06/11-2015
        -----------------------------------------------------------------------------------------------------------
        Note: The sgsim is capable of simulating 3-d fields, however, we have only given paramters for simulation of
              a 2-D field. Upgrading this
        """

        # Allocate for use when generating the realizations
        self.mean = mean
        self.outfile = outfile
        self.number = number

        self.sgsim_input = os.getcwd() + os.sep + 'sgsim.par'
        # write the sgsim input file following the outline given in the GSLIB book
        with open(self.sgsim_input, 'w') as file:
            file.write('\t\t\t Parameters for SGSIM \n')
            file.write('\t\t\t ******************** \n\n')
            file.write('START OF PARAMETERS:\n')
            file.write('{} \n'.format(data))    # file with data
            file.write('1 2 0 3 0 0 \n')        # Column number for x, y, and the variable in data file,
                                                # decluster weight, secondary variable (external drift)
            file.write('{} {} \n'.format(limits[0], limits[1]))  # tmin, and tmax. Values below or above are ignored
            file.write('0 \n')                  # 0: assume standard normal (no transfomation). 1: transform
            file.write('sgsim_rans.out \n')       # Output file for transformation table
            file.write('0 \n')                  # 0:data histogram used for transformation, 1: transformed according to
                                                # file (given in next key)
            file.write('sgsim_smth.in \n')         # File with transformation values
            file.write('1 2 \n')                # Columns for variable and weight in foosmth
            file.write('0.0 15.0 \n')           # min and max allowable data values
            file.write('1 0.0 \n')              # Interpolation in lower tail...
            file.write('1 15.0 \n')             # Interpolation in upper tail...
            file.write('0 \n')                  # Debugging level [0-3], 0 least debug info
            file.write('sgsim_debug.dbg \n')            # File with debug info
            file.write('{} \n'.format(outfile)) # file containing output-info
            file.write('{} \n'.format(number))  # number of simulations
            file.write('{} 0.5 1.0 \n'.format(x_size)) # define grid system along x axis
            file.write('{} 0.5 1.0 \n'.format(y_size)) # define grid system along y axis
            file.write('1 0.5 1.0 \n') # define grid system along z axis (Not implemented)
            seed_set = datetime.now().microsecond   # set seed given the time
            if seed_set%2 == 0: # Check if even, sgsim need odd integer
                seed_set += 1
            file.write('{} \n'.format(seed_set)) # Random number seed
            file.write('0 8 \n')                # min and max numb. data points used for each node
            file.write('20 \n')                 # maximum number of previously simulated nodes to use
            file.write('1 \n')                  # 0: data and simulated nodes searched separately, 1: they are combined
            file.write('0  0\n')                 # 0: standard spiral, search. 1 num: multiple grid. How many grids
            file.write('0 \n')                  # number of data pr. octant. If 0, not used
            file.write('{:1.1f} {:1.1f} 1.0 \n'.format(corr_range, corr_range*corr_aniso)) # search radius in maximum minimum
                                                # horizontal direction, and vertical (set to 1)
            file.write('{:1.1f} 0.0 0.0 \n'.format(corr_angl)) # orientation of search ellipse, rotation around y-axis.
                                                # 3-D rotation is not utilized yet.
            file.write('51 51 11 \n')           # Size of covariance lookup table
            file.write('0 0 0 \n')                  # Kriging type. 0:SK, 1:OK, 2:SK with locally varying mean, 3:K with
                                                # external drift, 4: Collocated cokriging with secondary variable.
                                                # 4 can be usefull if one simulated correlated fields
                                                # Corr coeff and var reduction for collocated cokriging
            file.write('bar.in \n')             # File for locally varying mean, external drift variable, or
                                                # secondary variable for cokriging.
            file.write('1 \n')                  # Column for secondary variable
            file.write('1 0 \n')                # Number of varigram structures (set to 1) and nugget constant
            if var_type == 'sph':
                var_ind = 1
            elif var_type == 'exp':
                var_ind = 2
            elif var_type == 'Gauss':
                var_ind = 3
            else:
                sys.exit('Plase define a valud varigram structure')
            file.write('{} {} {} 0.0 0.0 \n {:1.1f} {:1.1f} 1.0 \n'.format(var_ind, var, corr_angl, corr_range,
                                                                corr_range*corr_aniso)) # struct number, variogram, variance(sill),
                                                                          # anisotropi angles, correlation ranges

    def init_sisim(self, x_size, y_size, cat, thresh, var_type, var, cdf, data='foo.dat', cat_type=0, M_B=0,
                  limits=[0, 1], outfile='sisim.out', number=1, corr_range=[1], corr_aniso=[1], corr_angl=[0],
                   mean=None, facies_var=None):
        """
        This script writes the input file for gslib's sequential indicator simulation program.
        Input:
                - x_size: Size of field in x-direction
                - y_size: Size of field in y-direction
                - cat:    Number of categories
                - thresh: Threshold values (must be cat values and iterable)
                - cdf:    Global CDF or PDF values (must be cat values and iterable)
                - var_type: list of variogramtyps, as long as cat
                - var: Variance for the different categories, as long as cat
        Optional input:
                - Data:   File with data (if it does not exists => unconditional)
                - cat_type: variable typw (1=continous, 0=categorical)
                - M_B:    Markov-Bayes type simulation. 0=no, 1=yes
                - limits: trimming limits
                - outfile: string containing name of the file where data are stored
                - number: number of simulations
                - corr_range: correlation range in maximum horizontal direction
                - corr_aniso: Anisotropi factor.
                - corr_angl: Angle of primary correlation. Defined clockwise around the y-axis.

        """

        self.outfile = outfile
        self.sisim_input = os.getcwd() + os.sep + 'sisim.par'
        self.number = number
        self.mean = mean
        self.facies_var = facies_var

        with open(self.sisim_input, 'w') as file:
            file.write('\t\t\t Parameters for SISIM \n')
            file.write('\t\t\t ******************** \n\n')
            file.write('START OF PARAMETERS:\n')
            file.write('{}\n'.format(cat_type))      # 1=continous, 0=categorical
            file.write('{}\n'.format(cat))           # Numb theresholds/categories
            for val in thresh:
                file.write('{} '.format(val))          # Write the threshold values
            file.write('\n')
            for val in cdf:
                file.write('{} '.format(val))          # Write the cdf values
            file.write('\n')
            file.write('{}\n'.format(data))            # File with data
            file.write('1 2 0 3\n')                    # Columns for x,y,z and data
            file.write('sisim_soft.in\n')             # File with soft input (not used by us)
            file.write('1 2 0   3 4 5 6 7\n')         # Clumns for x,y,z and indicators
            file.write('{}\n'.format(M_B))    # Markov-Bayes simulation (0 = no, 1=yes)
            file.write('0.61 0.54 0.56 0.53\n') # calibration B(z) values, (if M_B = 1)
            file.write('{} {} \n'.format(limits[0], limits[1])) # trimming limits
            file.write('{} {} \n'.format(limits[0], limits[1])) # max and min data values
            file.write('1 0.0\n')             # extrapolation in the lower tail
            file.write('1 0.0\n')             # Extrapolation in the middle tail
            file.write('1 0.0\n')             # Extraoilation in the upper tail
            file.write('NA.dat\n')            # Values if # 3 is selected in the extrapolation
            file.write('3 0\n')               # Column values in file above
            file.write('0\n')                  # debug level, 0 is lowest detail
            file.write('sisim.dbg\n')          # Debug file
            file.write('{}\n'.format(outfile)) # file containing output
            file.write('{} \n'.format(number))  # number of simulations
            file.write('{} 0.5 1.0 \n'.format(x_size)) # define grid system along x axis
            file.write('{} 0.5 1.0 \n'.format(y_size)) # define grid system along y axis
            file.write('1 0.5 1.0 \n') # define grid system along z axis (Not implemented)
            seed_set = datetime.now().microsecond   # set seed given the time
            if seed_set%2 == 0: # Check if even, sgsim need odd integer
                seed_set += 1
            file.write('{} \n'.format(seed_set)) # Random number seed
            file.write('20\n')                  # max number of grid points used in simulation
            file.write('20\n')      # Max numb of previous nodes to use
            file.write('20\n')      # Max numb of soft dataa as node locations
            file.write('1\n')       # data are merged with grid nodes
            file.write('1\n')       # If set to 1, a multiple grid simulator is used
            file.write('5\n')       # Target numb of multgrid refinements
            file.write('0\n')       # Number of original data per octant
            file.write('{:1.1f} {:1.1f} 1.0 \n'.format(max(corr_range), max(corr_range)*max(corr_aniso))) # search radius in maximum minimum
                                                # horizontal direction, and vertical (set to 1)
            file.write('{:1.1f} 0.0 0.0 \n'.format(min(corr_angl))) # orientation of search ellipse, rotation around y-axis.
                                                # 3-D rotation is not utilized yet.
            file.write('51 51 11 \n')           # Size of covariance lookup table
            file.write('0 1\n')             # Full indicator kriging
            file.write('0\n')               # Simple kriging
            for i in range(cat):
                file.write('1 0 \n')                # Number of varigram structures (set to 1) and nugget constant
                if var_type[i] == 'sph':
                    var_ind = 1
                elif var_type[i] == 'exp':
                    var_ind = 2
                elif var_type[i] == 'Gauss':
                    var_ind = 3
                else:
                    sys.exit('Plase define a valud varigram structure')
                file.write('{} {} {} 0.0 0.0 \n {:1.1f} {:1.1f} 1.0 \n'.format(var_ind, var[i], corr_angl[i],
                                corr_range[i], corr_range[i]*corr_aniso[i])) # struct number, variogram, variance(sill),