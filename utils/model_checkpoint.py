import numpy as np

class ModelCheckpoint:
    def __init__(self, metric_fn, data_generator, num_sampled=10000):
        self.best_model = None
        self.best_metric = -np.inf
        self.metric_fn = metric_fn
        self.num_sampled = num_sampled
        self.data_generator = data_generator
    
    def __call__(self, model):
        X_val, y_val = self.data_generator(self.num_sampled)
        metric = self.metric_fn(y_val, model.predict(X_val))
        
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_model = model
            print("[ModelCheckpoint] New best model with %.5f validation accuracy" % metric)
        else:
            print("validation: %.3f accuracy" % metric)