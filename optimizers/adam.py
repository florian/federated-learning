import numpy as np

class Adam:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.alpha = 0.01
        self.eps = 1e-8
        
        self.t = 0
        
        self.m = None
        self.v = None
        self.theta = None
    
    def __call__(self, gradient):
        if self.t == 0:
            self.m = np.zeros(gradient.shape)
            self.v = np.zeros(gradient.shape)
            self.theta = np.zeros(gradient.shape)
        
        self.t += 1
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
        
        m_corrected = self.m / (1 - self.beta1**self.t)
        v_corrected = self.v / (1 - self.beta2**self.t)

        self.theta += self.alpha * m_corrected / (np.sqrt(v_corrected) + self.eps)
        
        return self.learning_rate * self.theta