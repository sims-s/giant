from process import Process
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import socket
import math
import time
from main import *
import winsound
from sklearn.linear_model import LogisticRegression

def easy_least_squares(driver):
    # paramaters
    num_global_samples = 100000
    dimension = 20
    num_workers = 1

    incoherent = False
    high_condition_num = True

    loss_type = LossFunctionType.QUADRATIC
    solve_type = SolverType.EXACT
    regularization_type = RegularizationType.NONE

    line_search = False

    max_itr = 8
    num_cg_steps = 10

    reg_mat = construct_reg_mat(regularization_type, dimension)
    weights = np.zeros(shape=[dimension, 1])
    
    driver.set_params(ADDRESS_LIST[:num_workers], num_workers, num_global_samples, weights, reg_mat, line_search, max_itr, verbose=True)

    # Make data matrix and partition it
    data_mat, labels = make_data_and_labels(num_global_samples, dimension, incoherent, high_condition_num, LabelsType.LINEAR_REGRESSION)
    partitioned_data, partitioned_labels = partition_data(data_mat, labels, num_workers)

    start = time.time()
    driver.start(partitioned_data, partitioned_labels, loss_type, solve_type, num_cg_steps)
    # send_workers_initial_data(address_list, partitioned_data, partitioned_labels, loss_type, solve_type, weights, reg_mat, num_global_samples, num_cg_steps)
    
    
    while not driver.done:
        pass
    winsound.Beep(2000, 400)
    print('RUN TIME WITH DATA: ', time.time() - start)
    print('RUN TIME WITHOUT DATA: ', time.time() - driver.alg_start_time)
    
    # print("buggy giant soln : ", driver.weights)
    # print('getting optimal solution...')
    optimal_soln = np.linalg.lstsq(data_mat, labels)[0]
    # print('optimal solution: ', optimal_soln)
    plot_relative_weight_error(optimal_soln, driver.weight_history)




def easy_ridge_regression(driver):
    # TODO: turn these into args so it's nicer
    num_global_samples = 1000
    dimension = 50
    num_workers = 32

    incoherent = True
    high_condition_num = False

    loss_type = LossFunctionType.QUADRATIC
    solve_type = SolverType.APPROXIMATE
    regularization_type = RegularizationType.IDENTITY
    regularization_constant = 10**-6
    line_search = False
    max_itr = 8
    num_cg_steps = 50

    

    # Make data matrix and partition it
    filepath = './datasets/YearPredictionMSD.txt'
    data_mat, labels = make_data_and_labels(num_global_samples, dimension, incoherent, high_condition_num, LabelsType.LINEAR_REGRESSION, filename=filepath)

    # initialize weights
    weights = np.zeros(shape=[data_mat.shape[1], 1])
    reg_mat = construct_reg_mat(regularization_type, data_mat.shape[1], regularization_constant)
    # Make a driver
    driver.set_params(ADDRESS_LIST[:num_workers], num_workers, data_mat.shape[0], weights, reg_mat, line_search, max_itr, verbose=False)
    partitioned_data, partitioned_labels = partition_data(data_mat, labels, num_workers)
    print('data is partitioned')
    
    start = time.time()
    driver.start(partitioned_data, partitioned_labels, loss_type, solve_type, num_cg_steps)
    
    while not driver.done:
        pass
    winsound.Beep(2000, 400)
    print(driver.weight_history)
    print('RUN TIME WITH DATA: ', time.time() - start)
    print('RUN TIME WITHOUT DATA: ', time.time() - driver.alg_start_time)
    # print("buggy giant soln : ", driver.weights)
    optimal_soln = np.linalg.inv(np.matmul(data_mat.T, data_mat) + regularization_constant*1/2*num_global_samples*np.eye(data_mat.shape[1]))
    optimal_soln = np.matmul(np.matmul(optimal_soln, data_mat.T), labels)
    # print('Optimal Solution: ', optimal_soln) 
    plot_relative_weight_error(optimal_soln, driver.weight_history)
    

def easy_logistic_regression(driver):
    # TODO: turn these into args so it's nicer
    num_global_samples = 100000
    dimension = 20
    num_workers = 12
    incoherent = False
    high_condition_num = False

    loss_type = LossFunctionType.LOGITISTC
    solve_type = SolverType.APPROXIMATE
    regularization_type = RegularizationType.IDENTITY
    regularization_constant = 10**-6
    line_search = False
    max_itr = 2
    num_cg_steps = 20

    reg_mat = construct_reg_mat(regularization_type, dimension, regularization_constant)

    # initialize weights
    weights = 140*np.ones(shape=[dimension, 1])
    # Make a driver
    driver.set_params(ADDRESS_LIST[:num_workers], num_workers, num_global_samples, weights, reg_mat, line_search, max_itr)

    # Make data matrix and partition it
    data_mat, labels = make_data_and_labels(num_global_samples, dimension, incoherent, high_condition_num, LabelsType.LOGISTIC_REGRESSION)


    partitioned_data, partitioned_labels = partition_data(data_mat, labels, num_workers)
    start = time.time()
    driver.start(partitioned_data, partitioned_labels, loss_type, solve_type, num_cg_steps)

    
    while not driver.done:
        pass
    winsound.Beep(2000, 400)
    # print('RUN TIME WITH DATA: ', time.time() - start)
    print('RUN TIME WITHOUT DATA: ', time.time() - driver.alg_start_time)
    
    # print("buggy giant soln : ", driver.weights)


    # # print('getting optimal solution...')
    logreg = LogisticRegression(C=1/(num_global_samples*regularization_constant), fit_intercept=False)
    logreg.fit(data_mat, np.ravel(np.maximum(labels, np.zeros(shape=labels.shape))))
    opt_weights = logreg.coef_
    # print('optimal solution : ', opt_weights)
    plot_relative_weight_error(opt_weights, driver.weight_history)








def fourier_least_squares(driver):
    # TODO: turn these into args so it's nicer
    num_global_samples = 1000
    dimension = 20
    num_workers = 2
    incoherent = True
    high_condition_num = False
    loss_type = LossFunctionType.QUADRATIC
    solve_type = SolverType.EXACT
    regularization_type = RegularizationType.NONE
    line_search = False
    max_itr = 10
    num_cg_steps = 10
    fourier_dim = 100

    reg_mat = construct_reg_mat(regularization_type, fourier_dim)

    # initialize weights
    weights = np.zeros(shape=[fourier_dim, 1])
    # Make a driver
    address_list = [('127.0.0.1', 5001+i) for i in range(num_workers)]
    driver.set_params(ADDRESS_LIST[:num_workers], num_workers, num_global_samples, weights, reg_mat, line_search, max_itr, verbose=False)

    # Make data matrix and partition it

    data_mat, labels = make_data_and_labels(num_global_samples, dimension, incoherent, high_condition_num, LabelsType.LINEAR_REGRESSION, fourier_dim=fourier_dim)


    partitioned_data, partitioned_labels = partition_data(data_mat, labels, num_workers)
    driver.start(partitioned_data, partitioned_labels, loss_type, solve_type, num_cg_steps)
    
    while not driver.done:
        pass
    winsound.Beep(2000, 400)
    
    print("buggy giant soln : ", driver.weights)
    # print('getting optimal solution...')
    optimal_soln = np.linalg.lstsq(data_mat, labels)[0]
    print('optimal solution: ', optimal_soln)
    plot_relative_weight_error(optimal_soln, driver.weight_history)
    
