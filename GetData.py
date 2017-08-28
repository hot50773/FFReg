'''
=================================================================================================
Getting 1d Simulation Data of Function-on-Function Finear Fegression
=================================================================================================
'''

import numpy as np


class Get_X_data(object):
    '''
    -----------------------------------------------------------
    input
    -----------------------------------------------------------
        Mean_Func : function
        Eigen_Func: function
        -------------------------------------------------------
        eigan_val : vector 
        -------------------------------------------------------
        domain    : length-2 vector (domain of time)  
        -------------------------------------------------------
        tran_size : int
        test_size : int
        num_pts   : int or length-2 vector
        num_grid  : int
        sparse : bool
    -----------------------------------------------------------
    output
    -----------------------------------------------------------
        time_grid: vector with length (num_grid)
        -------------------------------------------------------
        train_time_pts  : matrix (train_size * num_pts)
        obs_tran        : matrix (train_size * num_pts)
        -------------------------------------------------------
        test_time_pts   : matrix (test_size  * num_pts)
        obs_test        : matrix (test_size  * num_pts)
        -------------------------------------------------------
        real_val_on_grid: matrix (samp_size  * num_grid)
        -------------------------------------------------------
        mean_fns : matrix  (samp_size * num_pts) 
        fpcs     : matirx  (num_of_eigen * samp_size)
        eigen_fns: ndarray (num_of_eigen * samp_size * num_pts)
    -----------------------------------------------------------
    '''
    def __init__(self, Mean_Func, Eigen_Funcs, eigen_val, domain,
                 tran_size, test_size, num_grid, num_pts = None, epsilon_sd = 0.5, sparse = False):
        
        samp_size = tran_size + test_size
        num_pts = np.asarray(num_pts)
        num_grid = np.asarray(num_grid)
        
        #Generate time
        time_grid = np.linspace(domain[0], domain[1], num = num_grid)
        
        if sparse == True:    
            #sparse time
            if num_pts.size is 2:
                time_pts = np.random.uniform(domain[0], domain[1], (samp_size, num_pts[1]))
                condlist = np.hstack((np.ones((samp_size, num_pts[0]), dtype = np.bool), 
                                      np.random.choice([True,False], 
                                                       [samp_size, num_pts[1] - num_pts[0]],
                                                       replace=True)))
                
                time_pts = np.select([condlist], [time_pts], default = np.nan)
                
            elif num_pts.size is 1:
                time_pts = np.random.uniform(domain[0], domain[1], (samp_size, num_pts[0]))
                
            else:
                raise ValueError("argument 'num_pts' should be int or length-2 vector")
                
        else:
            #regular dense time
            time_pts = np.tile(time_grid, samp_size).reshape(samp_size, num_grid)
            num_pts = num_grid
            
        if num_pts.size is 1:
            #reshape num_pts
            num_pts = num_pts.repeat(2)
            
        #generate (mean fns, eigen fns, fpc scores, epsilon)
        mean_fns  = Mean_Func(time_pts)
        eigen_fns = Eigen_Funcs(time_pts)
        fpcs      = np.random.normal(0, np.sqrt(eigen_val), (samp_size, eigen_val.shape[0])).T
        epsilon   = np.random.normal(0, epsilon_sd, (samp_size, num_pts[1]))
        
        
        #generate simulation data
        
        real_val_of_data = np.zeros_like(time_pts)
        real_val_on_grid = np.zeros([samp_size, num_grid])
        
        for n_f in range(samp_size):
            real_val_of_data[n_f]  = mean_fns.take(n_f, 0) + fpcs.take(n_f, 1).dot(eigen_fns.take(n_f, 1))
            real_val_on_grid[n_f] =  Mean_Func(time_grid) + fpcs.take(n_f, 1).dot(Eigen_Funcs(time_grid))   

        #Seperate data into training set and test set, add epsilon
        obs_data = real_val_of_data + epsilon
        obs_tran = obs_data[:tran_size,:]
        obs_test = obs_data[tran_size:,:]
        
        tran_time_pts, test_time_pts = np.split(time_pts, [tran_size])
        
        #return results
        self.time_grid = time_grid
        self.tran_time_pts = tran_time_pts
        self.test_time_pts = test_time_pts
        self.obs_tran = obs_tran
        self.obs_test = obs_test
        self.real_val_on_grid = real_val_on_grid
        self.mean_fns = mean_fns
        self.eigen_fns = eigen_fns
        self.fpcs = fpcs

        
        
#-------------------------------------------------------------------------------------------------


class Get_Y_data(object):
    '''
    -----------------------------------------------------------
    input
    -----------------------------------------------------------
        Mean_Func : function
        Eigen_Func: function
        -------------------------------------------------------
        X_fpcs    : tuple
        B         : tuple
        -------------------------------------------------------
        domain    : length-2 vector (domain of time)  
        -------------------------------------------------------
        tran_size : int
        test_size : int
        num_pts   : int or length-2 vector
        num_grid  : int
    -----------------------------------------------------------
    output
    -----------------------------------------------------------
        time_grid: vector with length (num_grid)
        -------------------------------------------------------
        train_time_pts  : matrix (train_size * num_pts)
        obs_tran        : matrix (train_size * num_pts)
        -------------------------------------------------------
        test_time_pts   : matrix (test_size  * num_pts)
        obs_test        : matrix (test_size  * num_pts)
        -------------------------------------------------------
        real_val_on_grid: matrix (samp_size  * num_grid)
        -------------------------------------------------------
        mean_fns : matrix  (samp_size * num_pts) 
        fpcs     : matirx  (num_of_eigen * samp_size)
        eigen_fns: ndarray (num_of_eigen * samp_size * num_pts)
    -----------------------------------------------------------
    '''
    def __init__(self, Mean_Func, Eigen_Funcs, X_fpcs, B, domain,
                 test_size, num_grid, num_pts = None, epsilon_sd = 0.5, sparse = False):
        
        samp_size = X_fpcs[0].shape[1]
        tran_size = samp_size - test_size
        num_pts = np.asarray(num_pts)
        num_grid = np.asarray(num_grid)
        
        #Generate time
        time_grid = np.linspace(domain[0], domain[1], num = num_grid)
        if sparse == True:
            
            #sparse time
            if num_pts.size is 2:                
                time_pts = np.random.uniform(domain[0], domain[1], (samp_size, num_pts[1]))
                condlist = np.hstack((np.ones((samp_size, num_pts[0]), dtype = np.bool), 
                                      np.random.choice([True,False], 
                                                       [samp_size, num_pts[1] - num_pts[0]], replace=True)))
                
                time_pts = np.select([condlist], [time_pts], default = np.nan)
                
            elif num_pts.size is 1:
                time_pts = np.random.uniform(domain[0], domain[1], (samp_size, num_pts[0]))
                
            else:
                raise ValueError("argument 'num_pts' should be int or length-2 vector")
                
        else:
            #regular dense time
            time_pts = np.tile(time_grid, samp_size).reshape(samp_size, num_grid)
            num_pts = num_grid
            
        if num_pts.size is 1:
            #reshape num_pts
            num_pts = num_pts.repeat(2)

        #generate (mean fns, eigen fns)
        mean_fns  = Mean_Func(time_pts)
        eigen_fns = Eigen_Funcs(time_pts)
        
        #--------------------------------------------
        #get fpc score
        fpcs = np.zeros([B[0].shape[1], samp_size])
        for i in range(len(X_fpcs)):
            fpcs += np.matmul(B[i].T, X_fpcs[i])
        #--------------------------------------------
        
        #get random error
        epsilon   = np.random.normal(0, epsilon_sd, (samp_size, num_pts[1]))

        #generate simulation data
        real_val_of_data = np.zeros_like(time_pts)
        real_val_on_grid = np.zeros([samp_size, num_grid])
        
        for n_f in range(samp_size):
            real_val_of_data[n_f]  = mean_fns.take(n_f, 0) + fpcs.take(n_f, 1).dot(eigen_fns.take(n_f, 1))
            real_val_on_grid[n_f] =  Mean_Func(time_grid)  + fpcs.take(n_f, 1).dot(Eigen_Funcs(time_grid))   
        ###

        #Seperate data into traning set and test set, add epsilon for training set
        obs_data = real_val_of_data + epsilon
        obs_tran = obs_data[:tran_size,:]
        obs_test = obs_data[tran_size:,:]
        tran_time_pts, test_time_pts = np.split(time_pts, [tran_size])
        
        #get results
        self.time_grid = time_grid
        self.tran_time_pts = tran_time_pts
        self.test_time_pts = test_time_pts
        self.obs_tran = obs_tran
        self.obs_test = obs_test
        self.real_val_on_grid = real_val_on_grid
        self.mean_fns = mean_fns
        self.eigen_fns = eigen_fns
        self.fpcs = fpcs

        
#-------------------------------------------------------------------------------------------------
class Load_Data(object):
    '''
    -----------------------------------------------------------
    Load data from '.mat'
    -----------------------------------------------------------
    input
    -----------------------------------------------------------
        sim_data:{}
        time_grid: vector
        sim_data_names: variable names in sim_data
        tran_test_size: len-2 vector
        response: bool
    -----------------------------------------------------------
    output
    -----------------------------------------------------------
        time_grid: vector with length (num_grid)
        -------------------------------------------------------
        train_time_pts  : matrix (train_size * num_pts)
        obs_tran        : matrix (train_size * num_pts)
        -------------------------------------------------------
        test_time_pts   : matrix (test_size  * num_pts)
        obs_test        : matrix (test_size  * num_pts)
        -------------------------------------------------------
        real_val_on_grid: matrix (samp_size  * num_grid)
        -------------------------------------------------------
        fpcs     : matirx  (num_of_eigen * samp_size)
    -----------------------------------------------------------
    '''
    def __init__(self, sim_data, time_grid, sim_data_names = ['t', 'obs', 'fpcs', 'real_test_Y'],
                 tran_test_size = np.array([200, 100]), response = False):
        
        tran_size, test_size = tran_test_size
        samp_size = tran_size + test_size
        time_pts = sim_data[sim_data_names[0]]
        obs_data = sim_data[sim_data_names[1]]
        fpcs     = sim_data[sim_data_names[2]]
        num_grid = time_grid.size
        if response:
            real_val_on_grid = np.zeros([samp_size, num_grid])
            real_val_on_grid[tran_size:,] = sim_data[sim_data_names[3]]
        else:
            real_val_on_grid =np.nan
            
        #Seperate data into training set and test set
        obs_tran = obs_data[:tran_size,:]
        obs_test = obs_data[tran_size:,:]
        tran_time_pts, test_time_pts = np.split(time_pts, [tran_size])
        #return results
        self.time_grid = time_grid
        self.tran_time_pts = tran_time_pts
        self.test_time_pts = test_time_pts
        self.obs_tran = obs_tran
        self.obs_test = obs_test
        self.real_val_on_grid = real_val_on_grid
        self.fpcs = fpcs
