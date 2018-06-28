import numpy as np

class RProp:
    def __init__(self, learning_rate, num_features):
        self.learning_rates = np.full(num_features, learning_rate)
        self.min = 0.0000001
        self.max = 50
        self.a = 1.2
        self.b = 0.5
        
        self.t = 0
        self.last_gradient = None
    
    def __call__(self, gradient):
        if self.t >= 1:
            for i in range(len(gradient)):
                if gradient[i] * self.last_gradient[i] > 0:
                    self.learning_rates[i] = min(self.learning_rates[i] * self.a, self.max)
                elif gradient[i] * self.last_gradient[i] < 0:
                    self.learning_rates[i] = max(self.learning_rates[i] * self.b, self.min)
            
        self.t += 1
        self.last_gradient = gradient
        
        return self.learning_rates * np.sign(gradient)