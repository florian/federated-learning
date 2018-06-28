import numpy as np

class GradientDescent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        
    def __call__(self, gradient):
        return self.learning_rate * gradient
    
class AdaptiveGradientDescent:
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
        
        return self.learning_rates * gradient
    
class DecayedGradientDescent:
    def __init__(self, learning_rate, decay):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epoch = 0
        
    def __call__(self, gradient):
        result = self.learning_rate * gradient
        
        self.learning_rate = self.learning_rate * 1 / (1 + self.decay * self.epoch)
        self.epoch += 1
            
        return result