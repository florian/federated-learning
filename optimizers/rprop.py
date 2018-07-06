import numpy as np

class RProp:
    def __init__(self, learning_rate, num_features, min_value=0.0000001, max_value=50, alpha=1.2, beta=0.5):
        self.learning_rates = np.full(num_features, learning_rate)
        self.min = min_value
        self.max = max_value
        self.a = alpha
        self.b = beta
        
        self.t = 0
        self.last_gradient = None
    
    def __call__(self, gradient):
        if self.t >= 1:
            for i in range(len(gradient)):
                if gradient[i] * self.last_gradient[i] > 0:
                    self.learning_rates[i] = min(self.learning_rates[i] * self.a, self.max)
                elif gradient[i] * self.last_gradient[i] < 0:
                    self.learning_rates[i] = max(self.learning_rates[i] * self.b, self.min)
            
        self.learning_rates = np.round(self.learning_rates)
            
        self.t += 1
        self.last_gradient = gradient
        
        return self.learning_rates * np.sign(gradient)