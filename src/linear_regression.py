import jax
import jax.numpy as jnp

def model(params, cases):
    return jnp.array([jnp.dot(params[0:-1], case) + params[-1] for case in cases])

def loss_fn(params, feature_vectors, targets):
    preds = model(params, feature_vectors)
    loss = jnp.mean((preds - targets)**2)
    print(loss.primal)
    return loss

def update(loss_gradient, params, feature_vectors, targets, learning_rate):
    return params - learning_rate * loss_gradient(params, feature_vectors, targets)

def train(steps, params, feature_vectors, targets, learning_rate):
    loss_gradient = jax.grad(loss_fn) # Create gradient function
    for _ in range(steps):
        params = update(loss_gradient, params, feature_vectors, targets, learning_rate)
    return params

if __name__ == "__main__":
    # Data with 15 cases, 10 features, and 1 target
    feature_vectors = jnp.array([
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 2., 1., 2., 1., 2., 1., 2., 1., 2.],
        [1., 3., 1., 3., 1., 3., 1., 3., 1., 3.],
        [1., 4., 1., 4., 1., 4., 1., 4., 1., 4.],
        [1., 5., 1., 5., 1., 5., 1., 5., 1., 5.],
        [1., 6., 1., 6., 1., 6., 1., 6., 1., 6.],
        [1., 7., 1., 7., 1., 7., 1., 7., 1., 7.],
        [1., 8., 1., 8., 1., 8., 1., 8., 1., 8.],
        [1., 9., 1., 9., 1., 9., 1., 9., 1., 9.],
        [1., 10., 1., 10., 1., 10., 1., 10., 1., 10.],
        [1., 11., 1., 11., 1., 11., 1., 11., 1., 11.],
        [1., 12., 1., 12., 1., 12., 1., 12., 1., 12.],
        [1., 13., 1., 13., 1., 13., 1., 13., 1., 13.],
        [1., 14., 1., 14., 1., 14., 1., 14., 1., 14.],
        [1., 15., 1., 15., 1., 15., 1., 15., 1., 15.]
    ])
    
    targets = jnp.array([56., 81., 119., 22., 103., 57., 80., 118., 21., 104., 57., 82., 118., 22., 102.])

    params = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) # Initialize parameters

    # Hyperparameters
    learning_rate = 1e-3
    steps = 100

    # Train
    params = train(steps, params, feature_vectors, targets, learning_rate)
    print(params)