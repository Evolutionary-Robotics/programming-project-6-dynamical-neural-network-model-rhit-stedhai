import numpy as np

class CTRNN():
    def __init__(self, size):
        self.size = size                        
        self.states = np.zeros(size)            
        self.timeConstants = np.random.uniform(0.1,5.0,size=(self.size))    
        self.invTimeConstants = 1.0/self.timeConstants
        self.biases = np.random.uniform(-10,10,size=(self.size))        
        self.weights = np.random.uniform(-10,10,size=(self.size,self.size))
        self.inputs = np.zeros(size) 
        self.outputs = self.sigmoid(self.states+self.biases)          
                   
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def step(self, dt):
        currentInput = self.inputs + np.dot(self.weights.T, self.outputs)
        self.states += dt * (self.invTimeConstants*(-self.states+currentInput))
        self.outputs = self.sigmoid(self.states+self.biases)
    
    def save(self, filename):
        np.savez(filename, size=self.size, weights=self.weights, biases=self.biases, timeconstants=self.timeConstants)

    def load(self, filename):
        params = np.load(filename)
        self.size = params['size']
        self.weights = params['weights']
        self.biases = params['biases']
        self.timeConstants = params['timeconstants']
        self.invTimeConstants = 1.0/self.timeConstants

    def setTimeConstants(self, timeConstant):
        self.timeConstants = timeConstant
        self.invTimeConstants = 1.0/self.timeConstants

    def modifyTimeConstants(self, modifier):
        self.timeConstants *= modifier
        self.invTimeConstants = 1.0/self.timeConstants

    def showTimeConstants(self):
        print("timeConstants: ", self.timeConstants)
        print("invTimeConstants: ", self.invTimeConstants)
