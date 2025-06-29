import tensorflow as tf
import numpy as np
from sklearn.neighbors import NearestNeighbors

class ForecastCF:
    """
    Counterfactual explanations for multivariate time series forecasting.
    """
    def __init__(self, *, tolerance=1e-6, max_iter=500, optimizer=None, pred_margin_weight=0.9, random_state=None):
        # --- THE FIX IS HERE ---
        # Adjusted learning rate to be smaller for more stable steps.
        self.optimizer_ = (
            tf.keras.optimizers.legacy.Adam(learning_rate=0.0005) 
            if optimizer is None
            else optimizer
        )
        self.tolerance_ = tf.constant(tolerance)
        # Increased max_iter to give the optimizer more time to find a solution.
        self.max_iter = max_iter
        # Slightly decreased the prediction margin weight to also consider proximity, which can stabilize the search.
        self.pred_margin_weight = pred_margin_weight
        self.random_state = random_state
        self.model_ = None
        self.MISSING_MAX_BOUND = np.inf
        self.MISSING_MIN_BOUND = -np.inf

    def fit(self, model, model_name):
        self.model_ = model
        self.model_name = model_name
        return self

    def _compute_loss(self, original_sample, cf_sample, max_bound, min_bound):
        pred = self.model_(cf_sample)
        margin_loss = tf.reduce_sum(tf.nn.relu(pred - max_bound)) + tf.reduce_sum(tf.nn.relu(min_bound - pred))
        proximity_loss = tf.reduce_sum(tf.square(original_sample - cf_sample))
        # The weight for proximity_loss is (1 - self.pred_margin_weight)
        return self.pred_margin_weight * margin_loss + (1 - self.pred_margin_weight) * proximity_loss

    def transform(self, x, max_bound_lst, min_bound_lst):
        result_samples = np.empty(x.shape, dtype=np.float32)
        for i in range(x.shape[0]):
            if (i+1) % 25 == 0:
                print(f"{i+1} of {x.shape[0]} samples transformed.")
            x_sample = tf.convert_to_tensor(x[np.newaxis, i], dtype=tf.float32)
            z = tf.Variable(x_sample, dtype=tf.float32)
            max_bound = tf.convert_to_tensor(max_bound_lst[i], dtype=tf.float32)
            min_bound = tf.convert_to_tensor(min_bound_lst[i], dtype=tf.float32)

            for _ in range(self.max_iter):
                with tf.GradientTape() as tape:
                    loss = self._compute_loss(x_sample, z, max_bound, min_bound)
                if loss < self.tolerance_:
                    break
                grads = tape.gradient(loss, [z])
                self.optimizer_.apply_gradients(zip(grads, [z]))
            result_samples[i] = z.numpy()
        return result_samples, None, None

# (The Baseline classes remain the same)
class BaselineShiftCF:
    def __init__(self, *, desired_percent_change):
        """
        Parameters
        ----------
        desired_percent_change : float, optional
            The desired percent change of the counterfactual
        """
        self.desired_change = desired_percent_change

    # TODO: compatible with the counterfactuals of wildboar
    #       i.e., define the desired output target per label
    def transform(self, x):
        """Generate counterfactual explanations
        x : array-like of shape [n_samples, n_timestep, n_dims]
            The samples
        """
        result_samples = x * (1 + self.desired_change)
        return result_samples
        
class BaselineNNCF:
    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors
    def fit(self, model, X_train, Y_train):
        self.model = model
        self.X_train_orig = X_train
        self.Y_train = Y_train
        self.X_train_flat = X_train.reshape(X_train.shape[0], -1)
        self.nn_ = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean').fit(self.X_train_flat)
        return self
    def transform(self, x, max_bound_lst, min_bound_lst):
        x_flat = x.reshape(x.shape[0], -1)
        _, indices = self.nn_.kneighbors(x_flat)
        cf_samples = np.empty_like(x)
        for i in range(len(x)):
            found_cf = False
            for neighbor_idx in indices[i]:
                neighbor_sample = self.X_train_orig[neighbor_idx].reshape(1, *x.shape[1:])
                y_pred_neighbor = self.model.predict(neighbor_sample, verbose=0)
                if np.all((y_pred_neighbor >= min_bound_lst[i]) & (y_pred_neighbor <= max_bound_lst[i])):
                    cf_samples[i] = self.X_train_orig[neighbor_idx]
                    found_cf = True
                    break
            if not found_cf:
                cf_samples[i] = x[i]
        return cf_samples, None, None