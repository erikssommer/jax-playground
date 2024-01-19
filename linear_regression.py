import jax
import jax.numpy as jnp

def model(params, cases):
    return jnp.array([jnp.dot(params[0:-1], case) + params[-1] for case in cases])

def loss_fn(params, feature_vectors, targets):
    preds = model(params, feature_vectors)
    return jnp.mean((preds - targets)**2)

def update(loss_gradient, params, feature_vectors, targets, learning_rate):
    return params - learning_rate * loss_gradient(params, feature_vectors, targets)

def train(steps, params, feature_vectors, targets, learning_rate):
    loss_gradient = jax.grad(loss_fn) # Create gradient function
    for _ in range(steps):
        params = update(loss_gradient, params, feature_vectors, targets, learning_rate)
    return params

if __name__ == "__main__":
    # Data
    feature_vectors = jnp.array([[73., 67., 43., 12., 10.], 
                                 [91., 88., 64., 24., 12.], 
                                 [87., 134., 58., 33., 18.], 
                                 [102., 43., 37., 44., 19.], 
                                 [69., 96., 70., 12., 10.]], dtype=jnp.float32)
    targets = jnp.array([[56.], 
                         [81.], 
                         [119.], 
                         [22.], 
                         [103.]], dtype=jnp.float32)
    # Hyperparameters
    learning_rate = 1e-5
    steps = 200
    # Model parameters
    params = jnp.array([0.7, 0.3, 0.1, 0.5, 0.3, 0.1])
    # Train
    params = train(steps, params, feature_vectors, targets, learning_rate)
    print(params)