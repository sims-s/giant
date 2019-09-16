from process import Process
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import socket
import math
import time
from sklearn.linear_model import Ridge
from tests import *
import pickle
import os
from tqdm import tqdm
import csv

# From https://stackoverflow.com/questions/29798795/multivariate-student-t-distribution-with-python
def multivariate_t(mu,Sigma,N,M):
    '''
    Output:
    Produce M samples of d-dimensional multivariate t distribution
    Input:
    mu = mean (d dimensional numpy array or scalar)
    Sigma = scale matrix (dxd numpy array)
    N = degrees of freedom
    M = # of samples to produce
    '''
    d = len(Sigma)
    g = np.tile(np.random.gamma(N/2.,2./N,M),(d,1)).T
    Z = np.random.multivariate_normal(np.zeros(d),Sigma,M)
    return mu + Z/np.sqrt(g)


def make_data_and_labels(num_points, dimension, incoherent, high_condition_num, labels_type, fourier_dim = 0, make_new = False, filename=None):
    if filename:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            data = np.asarray(list(reader)).astype(np.float32)
            print(data.dtype)
            labels = data[:,0].reshape((-1,1))
            data = np.delete(data, 0, 1)
            return (data, labels)

    # If we don't want to make a new dataset, check to see if the data is saved
    if not make_new:
        # Find the folder it'd be in
        folder = '.\\datasets\\'
        folder += str(num_points) + 'x' + str(dimension) + '_'
        folder += 'inc_' if incoherent else 'coh_'
        folder += 'hcn\\' if high_condition_num else 'lcn\\'
        # Get the name of the labels file that we'd expect to find
        labels_name = 'labels'
        if labels_type==LabelsType.LINEAR_REGRESSION:
            labels_name += '_linear.npy'
        elif labels_type==LabelsType.LOGISTIC_REGRESSION:
            labels_name += '_logistic.npy'
        else:
            print("Unrecognized Label Type")
            raise NotImplementedError
        # See if the folder is there, and if it is
        if os.path.isdir(folder):
            print('loading data...')
            # Load the data
            # If we're not using the fourier'd data, then load it
            if not fourier_dim:
                data = np.load(folder + 'data.npy')
            # if we are, and it exists, then load it
            elif os.path.isfile(folder + 'data_fourier_%d.npy'%fourier_dim):
                fourier_data = np.load(folder + 'data_fourier_%d.npy'%fourier_dim)
            else:
            # otherwise construct it and save it
                print('constructing fourier features')
                fourier_data = construct_fourier_features(np.load(folder + 'data.npy'), fourier_dim)
                np.save(folder + 'data_fourier_%d.npy'%fourier_dim, fourier_data)
            # And if we've made labels for this type of problem, then load them
            if os.path.isfile(folder + labels_name):
                labels = np.load(folder + labels_name)
            # Otherwise, generate and save the labels of the appropriate type
            else:
                labels = make_labels(data, labels_type)
                # Then save the data
                np.save(folder + labels_name, labels)
            if not fourier_dim:
                print('Matrix coherence is: ', compute_coherence(data))
                return (data, labels)
            else:
                return (fourier_data, labels)
    # If we want to make new data or haven't made this particualr dataset yet
    # Generate the data
    print('generating data...')
    data = generate_synthetic_data(num_points, dimension, incoherent, high_condition_num)
    # And generate the labels
    labels = make_labels(data, labels_type)
    # And if we're not making a new dataset, then save it
    if not make_new:
        os.makedirs(folder)
        np.save(folder + 'data.npy', data)
        # If we're using fourier features, then construct them and save thems
        if fourier_dim:
            fourier_data = construct_fourier_features(data, fourier_dim)
            np.save(folder + 'data_fourier_%d.npy'%fourier_dim, fourier_data)
        np.save(folder + labels_name, labels)
    # If we don't want to save, but still want the fourier features, then do that
    elif fourier_dim:
        fourier_data = construct_fourier_features(data_mat, fourier_dim)
    if not fourier_dim:
        return (data, labels)
    else:
        return (fourier_data, labels)

def make_labels(data, labels_type):
    if labels_type==LabelsType.LINEAR_REGRESSION:
        return construct_labels_linear_reg(data)
    elif labels_type==LabelsType.LOGISTIC_REGRESSION:
        return construct_labels_logistic_reg(data)
    else:
        print("Unrecognized Label Type")
        raise NotImplementedError

def compute_coherence(data_matrix):
    U, S, V = np.linalg.svd(data_matrix, full_matrices=False)
    lvg_scores = []
    filter_through_dimension = 5
    # print(U.shape)
    U = U[:,:filter_through_dimension]
    for i in range(U.shape[0]):
        lvg_scores.append(np.linalg.norm(U[i,:])**2)
    return data_matrix.shape[0]/filter_through_dimension*max(lvg_scores)

# Generates synthetic data
def generate_synthetic_data(num_points, dimension, incoherent, high_condition_num):
    # Construct V as a basis for a stanard gaussian matrix
    random_gaussian = np.random.standard_normal(size=[dimension, dimension])
    Q, _ = np.linalg.qr(random_gaussian)
    V = Q
    # Create the mean and covariance matrix for making U
    means = np.ones(shape=[dimension])
    cov_mat = np.zeros(shape=[dimension, dimension])
    for i in range(dimension):
        for j in range(dimension):
            cov_mat[i,j]=2*.5**abs(i-j)
    
    U = None
    if incoherent:
        # If we want incoherent data, then use the basis of a random normal
        B = np.stack([np.random.multivariate_normal(mean = means, cov=cov_mat) for _ in range(num_points)])
        Q, _ = np.linalg.qr(B)
        U = Q
    else:
        # Otherwise, we use a student t with 4 degrees of freedom
        B = multivariate_t(means, cov_mat, 4, num_points)
        Q, _ = np.linalg.qr(B)
        U = Q
    singular_vals = None
    if high_condition_num:
        # If we want a high condition number, space out the singular values more
        singular_vals = np.asarray([10**i for i in np.linspace(0,-8,dimension)])
    else:
        # Otherwise don't space them out much
        singular_vals = np.asarray([10**i for i in np.linspace(0,-1,dimension)])
    # Then construct the data matrix and return it
    data = np.matmul(np.matmul(U, np.diag(singular_vals)), V.T)
    return data

# Constructs labels for linear regression
def construct_labels_linear_reg(data_mat):
    dimension = data_mat.shape[1]
    weights = np.ones(shape=[dimension])
    weights[int(.2*dimension):int(.8*dimension)]/=10
    noise = np.random.normal(0, .1**2, size=[data_mat.shape[0]])
    labels = np.matmul(data_mat, weights) + noise
    return labels.reshape((-1,1))

# Constructs labels for logistic regression
def construct_labels_logistic_reg(data_mat):
    dimension = data_mat.shape[1]
    weights = np.ones(shape=[dimension])
    weights[int(.2*dimension):int(.8*dimension)]/=10
    probabilities = np.exp(np.matmul(data_mat, weights))
    probabilities = np.divide(probabilities, (np.ones(shape=probabilities.shape) + probabilities)).reshape((-1,1))
    labels = np.random.binomial(1, probabilities)
    labels = 2*labels - np.ones(shape=labels.shape)
    return labels.reshape((-1,1))

# Fucntion that constructs the random fourier features
def construct_fourier_features(data_mat, dimension):
    print('computing sigam...')
    sigma = 1/data_mat.shape[0]**2*sum([np.linalg.norm(data_mat[i,:]-data_mat[j,:],2) for i in tqdm(range(data_mat.shape[0])) for j in range(data_mat.shape[0])])
    print('making R..')
    R = np.random.normal(loc=0, scale=1/sigma**2, size=[data_mat.shape[1], dimension])
    print('making q...')
    q = np.random.uniform(low=0, high=np.pi, size=dimension).reshape((-1,1))
    print('making the features')
    fourier_features = np.matmul(data_mat, R) + np.matmul(np.ones(shape=[data_mat.shape[0], dimension]), q)
    fourier_features = math.sqrt(2)*np.cos(fourier_features)
    return fourier_features


def partition_data(data_mat, labels, num_workers, num_local_samples=None):
    # first compute the number of local samples for each machine if it isn't given
    remainder = 0
    samples_per_worker = None
    num_global_samples = data_mat.shape[0]
    if (not num_local_samples):
        num_local_samples=math.floor(num_global_samples/num_workers)
        remainder = num_global_samples % num_workers
         # List with the number of samples that each worker should have
        samples_per_worker = [num_local_samples+1 if i<remainder else num_local_samples for i in range(num_workers)]
    else:
        samples_per_worker = [num_local_samples for i in range(num_workers)]
   
    # Then get the indecies that they all need to have in the data matrix
    indecies_for_workers = [sum(samples_per_worker[:i]) for i in range(num_workers+1)]
    while (indecies_for_workers[-1] > data_mat.shape[0]):
        data_mat = np.vstack((data_mat, data_mat))
        labels = np.vstack((labels, labels))
    # Shuffle the data just in case it came ordered
    stacked = np.hstack((data_mat, labels))
    np.random.shuffle(stacked)
    # Then get their data
    data_for_workers = [(stacked[indecies_for_workers[i]:indecies_for_workers[i+1],:-1],\
                         stacked[indecies_for_workers[i]:indecies_for_workers[i+1],-1].reshape((-1,1))) for i in range(num_workers)]
    partitioned_data = [partition[0] for partition in data_for_workers]
    partitioned_labels = [partition[1] for partition in data_for_workers]
    return (partitioned_data, partitioned_labels)

# Function that sends the workers their initial data
# Begins the whole GIANT Algorithm
def send_workers_initial_data(address_list, partitioned_data, partitioned_labels, loss_type, solve_type, initial_weights, reg_mat, num_global_samples, num_cg_steps):
    for address, data, labels in zip(address_list, partitioned_data, partitioned_labels):
        # Then make what we're sending
        msg_data = {"Data" : data, "Loss Function" : LossFunction(loss_type), "Solve Type" : solve_type, \
                    "Weights" : initial_weights, "Regularization Matrix" : reg_mat, \
                    "Labels" : labels, "Number of Global Samples" : num_global_samples, "Num CG Steps" : num_cg_steps}
        msg = Message(MessageType.SETUP_WORKER, HOME, msg_data)
        send_message(msg, address)
        # print('sent message')
        # time.sleep(1)



class Driver:
    def __init__(self, proc_address):
        self.proc_address = proc_address
        

        self.gradients_recieved = []
        self.directions_recieved = []
        self.loss_vals_recieved = []

        self.listening_thread = ListeningThread(self)
        self.listening_thread.daemon = True
        self.listening_thread.start()
        self.msg_handling_threads = []

    def set_params(self, worker_addresses, num_workers, num_global_samples, initial_weights, reg_mat, line_search, max_itr, verbose=False):
        self.worker_addresses = worker_addresses
        self.num_workers = num_workers
        self.alg_start_time = None
        self.alg_start = False

        self.weights = initial_weights
        self.weight_history = [self.weights]
        self.reg_mat = reg_mat
        self.line_search = line_search
        self.num_itr = 0
        self.max_itr = max_itr
        self.done = False
        self.num_global_samples = num_global_samples

        self.verbose = verbose


    def get_listen_address(self):
        return ('', self.proc_address[1])
        # return self.proc_address
    
    @staticmethod
    def _read_message(socket):
        """ Read a message from the socket."""
        # TODO make this receive an arbitrary size message
        msg_string = socket.recv(100 * 1024)
        return load_message_string(msg_string)

    def receive_gradient(self, gradient):
        if not self.alg_start:
            self.alg_start_time = time.time()
            self.alg_start = True
        self.gradients_recieved.append(gradient)
        # If we don't have all the gradients, then don't do anything
        if (len(self.gradients_recieved)<self.num_workers):
            # print("Only have %d gradients" %len(self.gradients_recieved))
            return
        # Otherwise aggregate them
        grad = np.sum(self.gradients_recieved, axis=0)
        self.gradient = grad + np.matmul(self.reg_mat, self.weights)
        # Reset the list of gradients
        self.gradients_recieved = []
        # Then broadcast them unless the gradient is super close to zero, in which case we stop early
        # to avoid problems with scipy's cg solve
        # print(self.gradient)
        if np.linalg.norm(self.gradient) < 10**-30:
            print("stopping early due to small gradient norm")
            self.done = True
            return
        msg = {"Gradient" : self.gradient}
        message_to_send = Message(MessageType.RECEIVE_GRADIENT_WORKER, self.get_listen_address(), msg)
        for address in self.worker_addresses:
            send_message(message_to_send, address)
    
    def receive_direction(self, direction):
        # First make the newton method
        self.directions_recieved.append(direction)
        if (len(self.directions_recieved)<self.num_workers):
            return
        self.newton_direction = 1/self.num_workers*np.sum(self.directions_recieved, axis=0)
        self.directions_recieved = []
        # If we're doing line search, then we need to send out the direciton
        if self.line_search:
            msg = {"Direction" : self.newton_direction}
            message_to_send = Message(MessageType.RECEIVE_DIRECTION_WORKER, self.get_listen_address(), msg)
            for address in self.worker_addresses:
                send_message(message_to_send, address)
            return
        self.weights = self.weights - self.newton_direction
        self.end_iteration()
        

    def select_step_size_by_backtracking(self, loss_list):
        function_value = loss_list.pop(0)
        index = 0

        while index < len(STEP_SIZES)-1 and \
            loss_list[index] > function_value + .1*STEP_SIZES[index]*np.dot(self.newton_direction.T, self.gradient):
            index+=1
        return STEP_SIZES[index]

    def add_regulariation_to_loss(self, loss_values):
        assert(len(loss_values)==1+len(STEP_SIZES))
        for i in range(len(loss_values)):
            if i==0:
                loss_values[i] += 1/2*np.matmul(np.matmul(self.weights.T, self.reg_mat), self.weights)
            else:
                step_index = i-1
                next_potential_weights = self.weights - STEP_SIZES[step_index]*self.newton_direction
                loss_values[i] += 1/2*np.matmul(np.matmul(next_potential_weights.T, self.reg_mat), next_potential_weights)
        return loss_values


    def receive_loss_vals(self, loss_vals):
        # Receive them
        self.loss_vals_recieved.append(loss_vals)
        if (len(self.loss_vals_recieved) < self.num_workers):
            return
        # Aggregate loss values if they're all here
        aggregated_loss_vals = [sum([losses[i] for losses in self.loss_vals_recieved]) for i in range(len(self.loss_vals_recieved[0]))]
        aggregated_loss_vals = self.add_regulariation_to_loss(aggregated_loss_vals)
        # select step size by backtracking
        step_size = self.select_step_size_by_backtracking(aggregated_loss_vals)
        self.weights = self.weights - step_size*self.newton_direction
        self.loss_vals_recieved = []
        self.end_iteration()

    def end_iteration(self):
        self.weight_history.append(self.weights)
        print("Finished with iteration %d" %self.num_itr)
        # print(self.weights)
        # print()
        self.num_itr+=1
        if (self.num_itr >= self.max_itr):
            self.done = True
            return
        msg = {"Weights" : self.weights}
        message_to_send = Message(MessageType.RECEIVE_WEIGHTS_WORKER, self.get_listen_address(), msg)
        for address in self.worker_addresses:
            send_message(message_to_send, address)


    def handle_message(self, message):
        msg_data = message.message_data
        # If we're recieiving gradietns
        if message.message_type==MessageType.RECEIVE_GRADIENT_DRIVER:
            if self.verbose:
                print("Driver receives the %dth gradient" %(len(self.gradients_recieved)+1))
            gradient = msg_data["Gradient"]
            self.receive_gradient(gradient)
        # If we're receiving directions
        elif message.message_type==MessageType.RECEIVE_DIRECTION_DRIVER:
            if self.verbose:
                print("Driver recieves the %dth direction" %(len(self.directions_recieved)+1))
            direction = msg_data["Direction"]
            self.receive_direction(direction)
        elif message.message_type==MessageType.RECEIVE_LOSS_VALUES_DRIVER:
            if self.verbose:
                print("Driver received the %dth set of loss values" %(len(self.loss_vals_recieved) + 1))
            loss_vals = msg_data["Loss Values"]
            self.receive_loss_vals(loss_vals)
        else:
            print("Unrecognized message type for driver!")
            raise NotImplementedError

    def start(self, partitioned_data, partitioned_labels, loss_type, solve_type, num_cg_steps):
        if self.verbose:
            print('sending data...')
        send_workers_initial_data(self.worker_addresses, partitioned_data, partitioned_labels, loss_type, solve_type, \
        self.weights, self.reg_mat, self.num_global_samples, num_cg_steps)
        if self.verbose:
            print('done sending data...')
            
                



def main():
    process_list = [Process(address, verbose=False) for address in ADDRESS_LIST]
    driver = Driver(HOME)
    driver.verbose=True
    easy_least_squares(driver)
    # easy_ridge_regression(driver)
    # easy_logistic_regression(driver)
    # fourier_least_squares(driver)

    # check out different ways of making a coherent matrix/compute the matrix coherence
    # try a real datasset?
    # look for other implementations of comparable methods to compare

    






if __name__ == "__main__":
    main()