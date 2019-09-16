import threading
from utils import *
import math
import numpy as np
import socket
import time
from scipy.sparse.linalg import cg
class Process:
    def __init__(self, proc_address, verbose=False):
        self.proc_address = proc_address
        self.listening_thread = ListeningThread(self)
        self.listening_thread.daemon = True
        self.listening_thread.start()
        self.msg_handling_threads = []

        self.hessian = None
        self.verbose = verbose
        

    def get_listen_address(self):
        return ('', self.proc_address[1])
        # return self.proc_address
    
    @staticmethod
    def _read_message(socket):
        """ Read a message from the socket."""
        # TODO make this receive an arbitrary size message
        ultimate_buffer = bytes()
        while True:
            receiving_buffer = socket.recv(1024)
            if not receiving_buffer: break
            ultimate_buffer+= receiving_buffer
        # msg_string = socket.recv(2**31)
        return load_message_string(ultimate_buffer)

    def recieve_init_data(self, msg_data):
        self.data = msg_data["Data"]
        # print(self.data.nbytes)
        self.num_local_samples = self.data.shape[0]
        self.loss = msg_data["Loss Function"]
        self.solve_type = msg_data["Solve Type"]
        self.weights = msg_data["Weights"]
        self.reg_mat = msg_data["Regularization Matrix"]
        self.labels = msg_data["Labels"]
        self.num_global_samples = msg_data["Number of Global Samples"]
        self.num_cg_steps = msg_data["Num CG Steps"]
    
    def __str__(self):
        to_return = '-------------------------------------------\n'
        to_return += 'Proc Address: Port %d' %self.get_listen_address()[1] + '\n'
        to_return += 'Data: \n' + str(self.data) + '\n'
        to_return += 'Labels: \n' + str(self.labels) + '\n'
        to_return += 'Num Local Samples: %d\n'%self.num_local_samples
        to_return += 'Num Global Samples: %d\n'%self.num_global_samples
        to_return += 'Loss Type: ' + str(self.loss.get_type())+ '\n'
        to_return += 'Solver Type: ' + str(self.solve_type)+ '\n'
        to_return += 'Current Weights: \n' + str(self.weights) + '\n'
        to_return += 'Regularization Matrix: \n' + str(self.reg_mat) + '\n'
        to_return += '-------------------------------------------\n'
        return to_return

    # Gets the hessian: returns if it's been computed before AND QUADRATIC
    # Computes it otherwise
    def get_hessian(self):
        if self.loss.get_type()==LossFunctionType.QUADRATIC:
            if not type(self.hessian) is np.ndarray:
                self.hessian = self.num_global_samples/self.num_local_samples*(np.matmul(self.A.T, self.A)) + self.reg_mat
            return self.hessian
        else:
            return self.num_global_samples/self.num_local_samples*np.matmul(self.A.T, self.A) + self.reg_mat
            

    # Computes and returns the gradient and sends it to the driver
    def return_gradient(self):
        gradient = self.comptue_gradient()
        msg_data = {"Gradient" : gradient}
        msg_to_send = Message(MessageType.RECEIVE_GRADIENT_DRIVER, self.get_listen_address(), msg_data)
        send_message(msg_to_send, HOME)

    def return_local_newton_direction(self):
        # Compute the local newton direction
        newton_direction = None
        if self.solve_type==SolverType.EXACT:
            newton_direction = self.compute_exact_newton_direction()
        elif self.solve_type==SolverType.APPROXIMATE:
            newton_direction = self.compute_approximate_newton_direction()
        else:
            print("Unrecognized Solve Type")
            raise NotImplementedError
        msg_data = {"Direction" : newton_direction}
        msg_to_send = Message(MessageType.RECEIVE_DIRECTION_DRIVER, self.get_listen_address(), msg_data)
        send_message(msg_to_send, HOME)
    
    def get_loss_given_weights(self, weights):
        # multiply each entry int he data matrix by the weights
        values = np.matmul(self.data, weights)
        # then sum up the loss function's value scaled by the reciporacla of the # fo data poitns
        loss = 1/self.num_global_samples * sum([self.loss.func_eval(values[i], self.labels[i]) for i in range(values.shape[0])])
        # Then add the regularaizaiton turm
        # Actually don't add it - add it in the driver otherwise will overestiamte
        # loss = loss + 1/2*np.matmul(np.matmul(weights.T, self.reg_mat), weights)
        return float(loss)


    def return_loss_evals_for_steps_in_direction(self, direction):
        step_sizes = [0] + STEP_SIZES
        loss_vals = []
        for step_size in step_sizes:
            loss_vals.append(self.get_loss_given_weights(self.weights - step_size*direction))
        msg_data = {"Loss Values" : loss_vals}
        msg_to_send = Message(MessageType.RECEIVE_LOSS_VALUES_DRIVER, self.get_listen_address(), msg_data)
        send_message(msg_to_send, HOME)


    def handle_message(self, message):
        msg_data = message.message_data
        # If we're doing setup
        if message.message_type==MessageType.SETUP_WORKER:
            if self.verbose:
                print("Worker %d recieves setup information" %self.get_listen_address()[1])
            # Then set the initial data
            self.recieve_init_data(msg_data)
            # And give the approximated gradient to the driver
            self.return_gradient()
        # If we're recieivng gradients
        elif message.message_type==MessageType.RECEIVE_GRADIENT_WORKER:
            if self.verbose:
                print("Worker %d recieves gradient" %self.get_listen_address()[1])
            self.aggregated_gradient = msg_data["Gradient"]
            # Send the exact local newton direction to the driver
            self.return_local_newton_direction()
            
        elif message.message_type==MessageType.RECEIVE_WEIGHTS_WORKER:
            if self.verbose:
                print("Worker %d recieves the weights" %self.get_listen_address()[1])
            new_weights = msg_data["Weights"]
            self.weights = new_weights
            self.return_gradient()
        elif message.message_type==MessageType.RECEIVE_DIRECTION_WORKER:
            if self.verbose:
                print("Worker %d recieves the direction" %self.get_listen_address()[1])
            search_direction = msg_data["Direction"]
            self.return_loss_evals_for_steps_in_direction(search_direction)
        else:
            raise NotImplementedError

    # Does factorized multiply for CG
    def hessian_multiply_for_CG(self, vec_to_multiply):
        hessian_multiply = np.matmul(self.A, vec_to_multiply)
        hessian_multiply = self.num_global_samples/self.num_local_samples*np.matmul(self.A.T, hessian_multiply)
        hessian_multiply += np.matmul(self.reg_mat, vec_to_multiply)
        return hessian_multiply

    # Does CG with an initial guess of Zero
    def compute_approximate_newton_direction(self):
        # Keep as false - my code is slower (irl time)
        # also my code seems to be a little less stable - sometimes things don't go quite right
        # same sort of thing happens with scipy cg, but happens less often & less bad
        my_cg = False
        # assert(my_cg)
        if not my_cg:
            return cg(self.get_hessian(), self.aggregated_gradient, tol=10**-16, maxiter=self.num_cg_steps)[0].reshape((-1,1))
        else:
            A = self.compute_A()
            curr_iterate = np.zeros(shape=self.aggregated_gradient.shape)
            residual = self.aggregated_gradient
            search_direction = self.aggregated_gradient
            for i in range(self.num_cg_steps):
                step_size = np.dot(residual.T, residual)/(np.dot(search_direction.T, self.hessian_multiply_for_CG(search_direction)))
                curr_iterate = curr_iterate + step_size*search_direction
                prev_residual = np.array(residual, copy=True)
                residual = residual - step_size*self.hessian_multiply_for_CG(search_direction)
                search_dir_step_size = np.dot(residual.T, residual)/np.dot(prev_residual.T, prev_residual)
                search_direction = residual + search_dir_step_size*search_direction
            return curr_iterate
        

    def compute_exact_newton_direction(self):
        hessian = self.get_hessian()
        # print('worker %d hessian: \n'%self.get_listen_address()[1], hessian)
        gradient = self.aggregated_gradient
        # print('worker %d gradient: \n'%self.get_listen_address()[1], gradient)
        local_newton_direction = np.matmul(np.linalg.inv(hessian), gradient)
        # print('worker %d direction: \n'%self.get_listen_address()[1], local_newton_direction)
        return local_newton_direction
    
    # Compute A as described in the paper
    def compute_A(self):
        A = math.sqrt(1/self.num_global_samples)*self.data
        for i in range(A.shape[0]):
            A[i,:]=A[i,:]*math.sqrt(self.loss.second_deriv_eval(np.dot(self.weights.T, A[i,:]), self.labels[i]))
        return A
    # Comptue B as described in the paper
    def compute_B(self):
        B = np.array([math.sqrt(1/self.num_global_samples)*self.loss.first_deriv_eval(np.dot(self.weights.T, self.data[i,:]), self.labels[i])/\
                        math.sqrt(self.loss.second_deriv_eval(np.dot(self.weights.T, self.data[i,:]), self.labels[i])) \
                                                                                            for i in range(self.data.shape[0])]).T
        return B        

    # Function that computes the gradient by constructing A and b
    def comptue_gradient(self):
        self.A = self.compute_A()
        B = self.compute_B().reshape((-1,1)).astype(np.float64)
        gradient = np.matmul(self.A.T, B)
        return gradient

    





        
