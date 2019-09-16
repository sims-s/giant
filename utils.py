from enum import Enum
import dill as pickle
import socket
import threading
import numpy as np
import matplotlib.pyplot as plt


aws = False

if aws:
    HOME = ('67.242.88.49', 5000)
    # TODO: CHANGE ME WHEN NEW SERVERS COME UP!!!
    ADDRESS_LIST = [("18.218.244.182", 5001)]
else:
    HOME = ('127.0.0.1', 5000)  
    ADDRESS_LIST = [('127.0.0.1', 5001+i) for i in range(64)]

STEP_SIZES = [4**i for i in range(0,-9,-1)]

class SolverType(Enum):
    EXACT = 0
    APPROXIMATE = 1

class LossFunctionType(Enum):
    QUADRATIC = 0
    LOGITISTC = 1

class RegularizationType(Enum):
    NONE = 0
    IDENTITY = 1

class LabelsType(Enum):
    LINEAR_REGRESSION = 0
    LOGISTIC_REGRESSION = 1

def plot_relative_weight_error(optimal_soln, weight_history):
    opt_norm = np.linalg.norm(optimal_soln, 2)
    optimal_soln = optimal_soln.reshape((-1,1))
    to_plot = [np.linalg.norm(optimal_soln-weight_history[i],2)/opt_norm for i in range(len(weight_history))]
    # print(to_plot)
    axes = plt.gca()
    # axes.get_yaxis().get_major_formatter().set_useOffset(False)
    plt.plot(to_plot)
    plt.title("lstsq_m=1_n=100k_d=100_s=5k_inc_hcn_exact")
    # plt.ylim((0,1))
    plt.xlabel("Iterations")
    plt.ylabel("Relative Error")
    plt.show()

def construct_reg_mat(type, dimension, reg_constant = None):
    if type==RegularizationType.NONE:
        return np.zeros(shape=[dimension, dimension])
    elif type==RegularizationType.IDENTITY:
        if reg_constant==None:
            return np.eye(dimension)
        return reg_constant*np.eye(dimension)
    else:
        print("Unrecognized Regularization Type!")
        raise NotImplementedError

class LossFunction():
    def __init__(self, _type):
        self.type = _type
        # self.z = sympy.symbols('z')
        # self.y = sympy.symbols('y')
        self.loss = None
        if self.type==LossFunctionType.QUADRATIC:
            self.loss = lambda z, y: (z-y)**2
            self.first_deriv = lambda z,y: 2*(z-y)
            self.second_deriv = lambda z,y: 2
        elif self.type==LossFunctionType.LOGITISTC:
            self.loss = lambda z,y: np.log(1+np.exp(-y*z))
            # TODO: CHECK THESE
            self.first_deriv = lambda z,y: -y/(1+np.exp(y*z))
            self.second_deriv = lambda z,y: np.exp(y*z)/((1+np.exp(y*z))**2)
        else:
            print("Unrecognized Loss Function!")
            raise NotImplementedError
        # self.first_deriv = sympy.diff(self.loss, self.z)
        # self.second_deriv = sympy.diff(self.first_deriv, self.z)

    def func_eval(self, value, label):
        return self.loss(value, label)
        # return self.loss.subs([(self.z, value), (self.y, label)])
    def first_deriv_eval(self, value, label):
        return self.first_deriv(value, label)
        # return self.first_deriv.subs([(self.z, value), (self.y, label)])
    def second_deriv_eval(self, value, label):
        return self.second_deriv(value, label)
        # return self.second_deriv.subs([(self.z, value), (self.y, label)])
    def get_type(self):
        return self.type



""" This class contains the message object which will be sent between
processes.
"""
class MessageType(Enum):
    TEST = -1
    SETUP_WORKER = 0
    RECEIVE_WEIGHTS_WORKER = 1
    RECEIVE_GRADIENT_WORKER = 2
    RECEIVE_GRADIENT_DRIVER = 3
    RECEIVE_DIRECTION_DRIVER = 4
    RECEIVE_DIRECTION_WORKER = 5
    RECEIVE_LOSS_VALUES_DRIVER = 6


class Message:
    def __init__(self, message_type, sender_address, message_data):
        self.message_type = message_type
        self.sender_address = sender_address
        self.message_data = message_data
        assert(self.message_type in MessageType)

    def __str__(self):
        """ Return a \"human readable\" string. """
        return ("Message from address %d of type %s with args %s" % 
            (self.sender_address[1], self.message_type, self.message_data))

def get_message_string(message):
    """ Return a string that can be sent over the network. """
    return pickle.dumps(message)


def load_message_string(string):
    """ Load a message string and return a message. """
    return pickle.loads(string)

def send_message(message, target_addr):
    """ Send a message over the socket. """
    # print("Sending message to Process @ %s" % str(target_addr))
    # print(message)

    msg_string = get_message_string(message)
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect(target_addr)
    soc.send(msg_string)
    soc.close()
    # print("sent message")

    
class MessageHandlingThread(threading.Thread):
    """ This thread is used by a process to handle messages."""
    def __init__(self, parent_process, message):
        threading.Thread.__init__(self)
        self.parent_process = parent_process
        self.message = message

    def run(self):
        self.parent_process.handle_message(self.message)




class ListeningThread(threading.Thread):
    def __init__(self, parent_process):
        threading.Thread.__init__(self)
        self.parent_process = parent_process

    def run(self):
        # print("LT: Starting to listen for incoming connections")

        # create a socket to listen to requests on
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.bind(self.parent_process.get_listen_address()) # socket.gethostname()
        soc.listen(10) # listen to up to 10 simultaneous connections

        while True:

            # print("LT: Waiting for next connection")
            
            # accept connections and handle them
            clientsocket, address = soc.accept()
            # print("LT: got connection from %s on port %d" % address)

            # read the message 
            message = self.parent_process._read_message(clientsocket)
            # print("LT: got message: %s" % str(message))
            # print("LT: got message")

            clientsocket.close()

            msg_thread = MessageHandlingThread(self.parent_process, message)
            msg_thread.daemon = True
            msg_thread.start()
            # put it here in case we want to access it later
            self.parent_process.msg_handling_threads.append(msg_thread)