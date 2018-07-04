'''
---------------------------------------------------------------------------------------------
this file contain functions for the simulation of function-on-function linear regression
---------------------------------------------------------------------------------------------

    1.Get_FPCA_Result (get results for predictor X)

    2.Fit_Mean_and_Cov(get results for response Y)

    3.Fit_Cov_XY      (fit Cov_Xi_Xj & Cov_Xi_Yj)

    4.Cv_Leave_One_Curve (function for Fit_Cov_XY)

    5.Real_Block_Cov_XX

    6.Real_Block_Cov_XY

    7.Real_Block_Eigen_Func

=============================================================================================
'''
import fpca
import lpr

###

import numpy as np
import scipy as sp
from scipy import interpolate
#=============================================================================================
def Get_FPCA_Result(X_data, candidate_h_mean, candidate_h_cov,
                  candidate_h_diag_cov, ker_fun = 'Gaussian', bw_select ='LeaveOneOut', fve = 0.85):

    num_grid  = X_data.time_grid.shape[0]
    tran_size = X_data.obs_tran.shape[0]
    num_pts = X_data.tran_time_pts.shape[1]

    result = fpca.Fpca(x = X_data.tran_time_pts.reshape(tran_size, num_pts, 1),
                       y = X_data.obs_tran,
                       x0 = X_data.time_grid.reshape(num_grid, 1),
                       h_mean = candidate_h_mean,
                       h_cov = candidate_h_cov,
                       h_cov_dia = candidate_h_diag_cov,
                       fve = fve,
                       ker_fun = ker_fun,
                       bw_select =bw_select)

    return(result)



def Cv_Leave_One_Curve(x, y, x0, candidate_h, ker_fun = 'Gaussian'):
    """
    input variable:
        x : matrix with dimension (num_of_points, dim)
        y : vecor with length num_of_points
        x0 : narray
        h : vector
        bin_width : vector
    """
    grid_shape = np.asarray(x0.shape[:-1])
    num_fun, num_pt, d = x.shape
    no_nan_val = ~np.isnan(y)
    bin_width = np.ptp(x0.reshape(-1, d), 0) / (grid_shape - 1)
    x0_min = x0.reshape(-1, d).min(0)
    x0_max = x0.reshape(-1, d).max(0)
    grid = tuple(np.linspace(x0_min.take(i), x0_max.take(i), x0.shape[i]) for i in range(d))

    bin_data_y, bin_data_num = lpr.Bin_Data(np.compress(no_nan_val.reshape(-1), x.reshape(-1, d), 0),
                                            np.compress(no_nan_val.reshape(-1), y.reshape(-1)),
                                            x0)
    #rand_num_fun = np.random.choice(num_fun, 100 if num_fun > 100 else num_fun, replace = False)
    rand_num_fun = np.arange(num_fun)
    test_fun = x.take(rand_num_fun, 0)
    test_y = y.take(rand_num_fun, 0)
    rand_num_no_nan_val = no_nan_val.take(rand_num_fun, 0)
    sse = np.ones(candidate_h.shape[0])
    for i in range(candidate_h.shape[0]):
        h = candidate_h.take(i, 0)
        r = lpr.Get_Range(bin_width, h, ker_fun)
        delta_x = lpr.Get_Delta_x(bin_width, r)
        weight = lpr.Get_Weight(delta_x, h, ker_fun)
        big_x = np.hstack((np.ones((delta_x.shape[0], 1)), delta_x))
        h_too_small = False
        for fun_i in range(test_fun.shape[0]):
            test_funi = test_fun.take(fun_i, 0)
            test_yi = test_y.take(fun_i, 0)
            fun_no_nan_value = rand_num_no_nan_val.take(fun_i, 0)
            fun_biny, fun_binx = lpr.Bin_Data(np.compress(fun_no_nan_value, test_funi, 0),
                                              np.compress(fun_no_nan_value, test_yi), x0)
            biny = bin_data_y - fun_biny
            binx = bin_data_num - fun_binx
            ext_biny, ext_binx = lpr.Extend_Bin_Data([biny, binx], r)
            train_y = lpr.Get_Linear_Solve(big_x.T, weight, ext_binx, ext_biny, r)
            if np.any(np.isnan(train_y)):
                h_too_small = True
                break
            interp_fun = sp.interpolate.RegularGridInterpolator(grid, train_y.reshape(x0.shape[:-1]),
                                                                bounds_error = False)
            interp_y = interp_fun(test_funi)
            sse[i] += np.nansum((test_yi - interp_y)**2)
        if h_too_small:
            sse[i] = np.nan
    opt_h = candidate_h.take(np.nanargmin(sse), 0)
    return(opt_h)

#=============================================================================================
def Fit_Cov_XY(X_time_pts, Y_time_pts, obs_X, obs_Y, X_time_grid, Y_time_grid,
               fit_X_mean, fit_Y_mean, candidate_h_cov, ker_fun = 'Gaussian'):
    '''
    ------------------------------------
    input
        --------------------------------
        X_time_pts: (n * k1) matrix
        Y_time_pts: (n * k2) matrix
        --------------------------------
        obs_X: (n * k1) matrix
        obs_Y: (n * k2) matrix
        --------------------------------
        X_time_grid: (length-m1) vector
        Y_time_grid: (length-m2) vector
        --------------------------------
        fit_X_mean:  (length-m1) vector
        fit_Y_mean:  (length-m2) vector
        --------------------------------
        candidate_h_cov: matrix
    ------------------------------------
    output
        --------------------------------
        fit_cov_XY: (m1 * m2) matrix
    ------------------------------------

    '''

    if X_time_pts.shape[0] != Y_time_pts.shape[0]:
        raise ValueError("sample size of X_time_pts and Y_time_pts must be the same !")
    if obs_X.shape[0] != obs_Y.shape[0]:
        raise ValueError("sample size of obs_X and obs_Y must be the same !")
    if obs_X.shape[0] != X_time_pts.shape[0]:
        raise ValueError("sample size of obs_X and X_time_pts must be the same !")

    samp_size, X_num_pts = X_time_pts.shape
    Y_num_pts = Y_time_pts.shape[1]

    time_of_XY = (np.vstack((np.repeat(X_time_pts, Y_num_pts).reshape(-1),
                             np.tile(Y_time_pts, X_num_pts).reshape(-1))).T)
    XY = np.einsum('ij,ik->ijk', obs_X, obs_Y).reshape(-1)
    time_grid_of_XY = np.asarray(np.meshgrid(X_time_grid, Y_time_grid)).T

    XY_bw = Cv_Leave_One_Curve(x  = time_of_XY.reshape(samp_size, X_num_pts * Y_num_pts, 2),
                               y  = XY.reshape(samp_size, -1),
                               x0 = time_grid_of_XY,
                               candidate_h = candidate_h_cov,
                               ker_fun = ker_fun)
    print('Bandwidth of cov: ', XY_bw)
    fit_XY = lpr.Lpr(x = time_of_XY, y = XY, x0 = time_grid_of_XY, h = XY_bw, ker_fun = ker_fun)
    fit_cov_XY = (fit_XY.reshape(X_time_grid.shape[0], Y_time_grid.shape[0])
                  - np.outer(fit_X_mean, fit_Y_mean))

    return(fit_cov_XY)

#=============================================================================================

class Fit_Mean_and_Cov(object):
    def __init__(self, X_data, h_mean, h_cov, binning = True, bin_weight = True,
                 ker_fun = 'Gaussian'):

        num_grid  = X_data.time_grid.shape[0]
        tran_size = X_data.obs_tran.shape[0]
        num_pts = X_data.tran_time_pts.shape[1]

        x = X_data.tran_time_pts.reshape(tran_size, num_pts, 1)
        y = X_data.obs_tran
        x0 = X_data.time_grid.reshape(num_grid, 1)

        self.__Check_Input(x, y, x0, h_mean, h_cov, binning, bin_weight, ker_fun)
        self.__num_fun, self.__num_pt, self.__d = tran_size, num_pts, 1
        self.__grid_shape = np.asarray(x0.shape[:-1])
        self.__bin_width = np.ptp(x0.reshape(-1, self.__d), 0) / (self.__grid_shape - 1)
        self.__n_train = lpr.Partition_Data_Size(self.__num_fun * self.__num_pt - np.isnan(y).sum())
        x0_min = x0.reshape(-1, self.__d).min(0)
        x0_max = x0.reshape(-1, self.__d).max(0)
        self.__grid = tuple(np.linspace(x0_min.take(i), x0_max.take(i), x0.shape[i]) for i in range(self.__d))
        self.__Main(x, y, x0, h_mean, h_cov, binning, bin_weight, ker_fun)
        self.__Standardize_Output()

    def __str__(self):
        print("Number of grid: %s" %" * ".join(map(str, self.__grid_shape)))
        print("Number of random function: %d" %self.__num_fun)
        # print("Number of points for each random function: %d" %self.__num_pt)
        print("="*20)
        print("Bandwidth of mean: [%s]" %", ".join(map(str, self.mean_bw)))
        print("Bandwidth of cov: [%s]" %", ".join(map(str, self.cov_bw)))
        return("="*20)

    def __Check_Input(self, x, y, x0, h_mean, h_cov, binning, bin_weight, ker_fun):

        if len(x.shape) is not 3:
            raise ValueError("Input x must be 3-d.(functions, points, dimension)")
        if len(y.shape) is not 2:
            raise ValueError("Input y must be 2-d.(functions, points)")
        if len(x0.shape[:-1]) is not x0.shape[-1]:
            raise ValueError("Input x0 dimension error.")
        if len(h_mean.shape) is not 2:
            raise ValueError("Input h_mean must be 2-d.")
        if len(h_cov.shape) is not 2:
            raise ValueError("Input h_mean must be 2-d.")
        if type(binning) is not bool:
            raise TypeError("Type of binning must be bool.")
        if type(bin_weight) is not bool:
            raise TypeError("Type of binning must be bool.")
        if ker_fun not in ['Gaussian', 'Epan']:
            raise ValueError("ker_fun should be 'Gaussian' or 'Epan'.")

    def __Cv_Leave_One_Curve(self, x, y, x0, candidate_h, ker_fun):
        """
        input variable:
            x : matrix with dimension (num_of_points, dim)
            y : vecor with length num_of_points
            x0 : narray
            h : vector
            bin_width : vector
        """
        n_grid = np.array(x0.shape[:-1])
        d = x0.shape[-1]
        no_nan_val = ~(np.isnan(y) | np.isnan(x).any(2))
        bin_data_y, bin_data_num = lpr.Bin_Data(np.compress(no_nan_val.reshape(-1), x.reshape(-1, d), 0), np.compress(no_nan_val.reshape(-1), y.reshape(-1)), x0)
        #rand_num_fun = np.random.choice(self.__num_fun, 100, replace = False)
        rand_num_fun = np.arange(self.__num_fun)
        test_fun = x.take(rand_num_fun, 0)
        test_y = y.take(rand_num_fun, 0)
        rand_num_no_nan_val = no_nan_val.take(rand_num_fun, 0)
        sse = np.ones(candidate_h.shape[0])
        for i in range(candidate_h.shape[0]):
            h = candidate_h.take(i, 0)
            r = lpr.Get_Range(self.__bin_width, h, ker_fun)
            delta_x = lpr.Get_Delta_x(self.__bin_width, r)
            weight = lpr.Get_Weight(delta_x, h, ker_fun)
            big_x = np.hstack((np.ones((delta_x.shape[0], 1)), delta_x))
            h_too_small = False
            for fun_i in range(test_fun.shape[0]):
                test_funi = test_fun.take(fun_i, 0)
                test_yi = test_y.take(fun_i, 0)
                fun_no_nan_value = rand_num_no_nan_val.take(fun_i, 0)
                fun_biny, fun_binx = lpr.Bin_Data(np.compress(fun_no_nan_value, test_funi, 0),
                                                  np.compress(fun_no_nan_value, test_yi), x0)
                biny = bin_data_y - fun_biny
                binx = bin_data_num - fun_binx
                ext_biny, ext_binx = lpr.Extend_Bin_Data([biny, binx], r)
                train_y = lpr.Get_Linear_Solve(big_x.T, weight, ext_binx, ext_biny, r)
                if np.any(np.isnan(train_y)):
                    h_too_small = True
                    break
                interp_fun = sp.interpolate.RegularGridInterpolator(self.__grid,
                                            train_y.reshape(x0.shape[:-1]), bounds_error = False)
                interp_y = interp_fun(test_funi)
                sse[i] += np.nansum((test_yi - interp_y)**2)
            if h_too_small:
                sse[i] = np.nan
        opt_h = candidate_h.take(np.nanargmin(sse), 0)
        return(opt_h)

    def __CV_Cov_Leave_One_Out(self, x, y, x0, candidate_h, ker_fun):
        grid_shape, d = np.asarray(x0.shape[:-1]), x0.shape[-1]
        n_grid = np.prod(self.__grid_shape)
        x_displacement = np.rint((x - x0.reshape(-1, d).min(axis = 0)) / self.__bin_width).astype(np.int32)
        x_p = np.sum(x_displacement * np.append(grid_shape[::-1].cumprod()[-2::-1], 1), axis = 2)
        xx_p = (x_p.repeat(x_p.shape[1], 1) * n_grid + np.tile(x_p, x_p.shape[1])).reshape(-1)
        yy = np.einsum('ij,ik->ijk', y, y).reshape(-1)
        random_order = np.random.choice(self.__num_fun,
                                        100 if self.__num_fun > 100 else self.__num_fun,
                                        replace=False)
        random_order = np.arange(self.__num_fun)
        test_fun = x.take(random_order, 0)
        test_xx_p = xx_p.reshape(self.__num_fun, -1).take(random_order, 0)
        test_yy = yy.reshape(self.__num_fun, -1).take(random_order, 0)
        non_nan_value = ~np.isnan(yy)
        xx_p = np.compress(non_nan_value, xx_p)
        yy = np.compress(non_nan_value, yy)
        tot_binx = np.bincount(xx_p, minlength = n_grid**2)
        tot_biny = np.bincount(xx_p, yy, minlength = n_grid**2)
        sse = np.ones(candidate_h.shape[0])
        for i in range(candidate_h.shape[0]):
            h = candidate_h.take(i, 0)
            r = lpr.Get_Range(self.__bin_width, h, ker_fun)
            delta_x = lpr.Get_Delta_x(self.__bin_width, r)
            weight = lpr.Get_Weight(delta_x, h, ker_fun)
            big_x = np.hstack((np.ones((delta_x.shape[0], 1)), delta_x))
            h_too_small = False
            for fun_i in range(test_xx_p.shape[0]):
                test_funi = test_fun.take(fun_i, 0)
                fun_non_nan_value = non_nan_value.reshape(self.__num_fun, -1).take(random_order.take(fun_i), 0)
                fun_xx = np.compress(fun_non_nan_value, test_xx_p.take(fun_i, 0))
                fun_yy = np.compress(fun_non_nan_value, test_yy.take(fun_i, 0))
                fun_binx = np.bincount(fun_xx, minlength = n_grid**2)
                fun_biny = np.bincount(fun_xx, fun_yy, minlength = n_grid**2)
                binx = tot_binx - fun_binx
                biny = tot_biny - fun_biny
                ext_biny, ext_binx = lpr.Extend_Bin_Data([biny.reshape(n_grid, n_grid), binx.reshape(n_grid, n_grid)], r)
                train_yy = lpr.Get_Linear_Solve(big_x.T, weight, ext_binx, ext_biny, r)
                if np.any(np.isnan(train_yy)):
                    h_too_small = True
                    break
                interp_fun = sp.interpolate.RegularGridInterpolator(self.__grid * 2, train_yy.reshape(x0.shape[:-1] * 2), bounds_error = False)
                interp_y = interp_fun(np.hstack((test_funi.repeat(self.__num_pt, 0),
                                      np.tile(test_funi.reshape(-1), self.__num_pt).reshape(-1, self.__d))))
                sse[i] += np.nansum((test_yy.take(fun_i, 0) - interp_y)**2)
            if h_too_small:
                sse[i] = np.nan
        h_opt = candidate_h.take(np.nanargmin(sse), 0)
        return([h_opt, xx_p, yy])

    def __Fit_Mean(self, x, y, x0, candidate_h, binning, bin_weight, ker_fun):
        self.mean_bw = self.__Cv_Leave_One_Curve(x, y, x0, candidate_h, ker_fun)

        self.mean_fun = lpr.Lpr(x = x.reshape(-1, self.__d),
                                y = y.reshape(-1),
                               x0 = x0,
                                h = self.mean_bw,
                                binning = binning,
                                bin_weight = bin_weight,
                                ker_fun = ker_fun)

    def __Fit_Cov(self, x, y, x0, candidate_h, binning, ker_fun):
        self.cov_bw, xx_p, yy = self.__CV_Cov_Leave_One_Out(x, y, x0, candidate_h, ker_fun)
        bin_xx = np.bincount(xx_p.reshape(-1), minlength = np.prod(self.__grid_shape)**2)
        bin_yy = np.bincount(xx_p.reshape(-1), yy.reshape(-1), minlength = np.prod(self.__grid_shape)**2)
        fit_yy = lpr.Lpr_For_Bin([bin_yy.reshape(np.tile(self.__grid_shape, 2)), bin_xx.reshape(np.tile(self.__grid_shape, 2))],
                                 np.tile(self.__bin_width, 2),
                                 h = self.cov_bw,
                                 ker_fun = ker_fun)
        self.cov_fun = fit_yy.reshape(np.repeat(np.prod(self.__grid_shape), 2)) - np.outer(self.mean_fun, self.mean_fun)

    def __Standardize_Output(self):
        self.mean_fun = self.mean_fun.reshape(self.__grid_shape)
        self.cov_fun = self.cov_fun.reshape(np.tile(self.__grid_shape, 2))

    def __Main(self, x, y, x0, h_mean, h_cov, binning, bin_weight, ker_fun):
        """
        input:
            x: ndarray (#functions, #points, #dimension)
            y: ndarray (#functions, #points)
            x0: ndarray
            candidate_h: ndarray (#h, #dimension)
            num_grid: vector
        """
        ###
        self.__Fit_Mean(x, y, x0, h_mean, binning, bin_weight, ker_fun)
        ###
        self.__Fit_Cov(x, y, x0, h_cov, binning, ker_fun)



#=============================================================================================

def Real_Block_Cov_XX(list_time_grid_X, list_X_real_val_on_grid, list_X_Mean_Func):
    '''
    input
        list_time_grid_X: list or tuple
        list_X_real_val_on_grid: list or tuple
        list_X_Mean_Func: list or tuple

    output
        real_block_cov_XX: matrix
    '''

    # create empty matrix
    real_XX = np.zeros(np.prod(np.asarray(list_time_grid_X).shape).repeat(2))
    num_of_X = len(list_X_real_val_on_grid)
    list_X_mean_funs = []

    for i in range(num_of_X):
        # create mean_fun_Xi on grid
        mean_fun_Xi = list_X_Mean_Func[i](list_time_grid_X[i])
        # add it to list
        list_X_mean_funs.append(mean_fun_Xi)
        # create the index for inserting real_Xi_Xj into real_XX
        num_grid_Xi = list_time_grid_X[i].size
        if i == 0:
            index_range = np.array([0, num_grid_Xi])
            print(index_range)
        else:
            index_range[0] = index_range[1]
            index_range[1] = index_range[1] + num_grid_Xi


        # create real Xi_Xj  (with j = i, i+1, ...)
        raw_Xi_Xj = np.einsum('ij,ik->ijk',list_X_real_val_on_grid[i], list_X_real_val_on_grid[i])
        real_Xi_Xj = np.mean(raw_Xi_Xj , axis= 0)
        for j in range(i + 1, num_of_X):

            raw_Xi_Xj_add = np.einsum('ij,ik->ijk',list_X_real_val_on_grid[i], list_X_real_val_on_grid[j])
            real_Xi_Xj_add = np.mean(raw_Xi_Xj_add, axis = 0)

            real_Xi_Xj = np.hstack((real_Xi_Xj, real_Xi_Xj_add))

        # insert real Xi_Xj into real_XX (only upper triangle)
        real_XX[index_range[0]:index_range[1], index_range[0]:] = real_Xi_Xj

    # fill lower traingle of real_XX
    up_tria_rm_diag   = np.triu(real_XX, 1)
    up_tria_with_diag = np.triu(real_XX, 0)
    real_XX = up_tria_rm_diag + up_tria_with_diag.T

    # combine all Xi_mean into a vector
    combine_X_mean = np.asanyarray(list_X_mean_funs).reshape(-1)
    real_block_cov_XX = real_XX - np.outer(combine_X_mean, combine_X_mean)
    return(real_block_cov_XX)
#=============================================================================================
def Real_Block_Cov_XY(list_time_grid_X, list_time_grid_Y, list_X_real_val_on_grid,
                      list_Y_real_val_on_grid, list_X_Mean_Func, list_Y_Mean_Func):
    '''
    all inputs: list or tuple

    output: matrix
    '''

    num_of_X = len(list_X_real_val_on_grid)
    num_of_Y = len(list_Y_real_val_on_grid)

    list_X_mean_funs = []
    list_Y_mean_funs = []

    # create empty matrix
    real_XY = np.zeros(np.prod(np.asarray(list_time_grid_X).shape),
                       np.prod(np.asarray(list_time_grid_Y).shape))

    for i in range(num_of_X):
        # create mean_fun_Xi on grid
        mean_fun_Xi = list_X_Mean_Func[i](list_time_grid_X[i])
        # add it to list
        list_X_mean_funs.append(mean_fun_Xi)

        # create real Xi_Xj  (with j = i, i+1, ...)
        raw_Xi_Yj = np.einsum('ij,ik->ijk',list_X_real_val_on_grid[i], list_Y_real_val_on_grid[0])
        real_Xi_Yj = np.mean(raw_Xi_Yj , axis= 0)
        for j in range(num_of_Y):
            if i == 0:
                #create mean fun_Yj on grid
                mean_fun_Yj = list_Y_Mean_Func[j](list_time_grid_Y[j])
                # add it to list
                list_Y_mean_funs.append(mean_fun_Yj)

            if num_of_Y > 1:
                # create real_Xi_Yj (for j = 0, ..., num_Y)
                raw_Xi_Yj_add = np.einsum('ij,ik->ijk',list_X_real_val_on_grid[i],
                                          list_Y_real_val_on_grid[j])
                real_Xi_Yj_add = np.mean(raw_Xi_Yj_add, axis = 0)
                real_Xi_Yj = np.hstack((real_Xi_Yj, real_Xi_Yj_add))

        # insert real Xi_Yj into real_XY (only upper triangle)
        if i == 0:
            real_XY = real_Xi_Yj
        else:
            real_XY = np.vstack((real_XY, real_Xi_Yj))

    # combine all Xi_mean into a vector
    combine_X_mean = np.asanyarray(list_X_mean_funs).reshape(-1)
    combine_Y_mean = np.asanyarray(list_Y_mean_funs).reshape(-1)
    # get real_block_cov_XY
    real_block_cov_XY = real_XY - np.outer(combine_X_mean, combine_Y_mean)
    return(real_block_cov_XY)

#=============================================================================================

def Real_Block_Eigen_Func(list_Eigen_Func, list_time_grid):
    if len(list_time_grid) == 1:
        return(list_Eigen_Func[0](list_time_grid[0]))
    else:
        real_block_eigen_func = list_Eigen_Func[0](list_time_grid[0])
        for i in range(1, len(list_time_grid)):
            real_block_eigen_func = sp.linalg.block_diag(real_block_eigen_func,
                                                         list_Eigen_Func[i](list_time_grid[i]))
        return(real_block_eigen_func)
