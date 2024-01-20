import numpy as np
import jax
import jax.numpy as jnp

# Feed forward neural network with 5 inputs, 10 hidden nodes, and 5 outputs with JAX

def gen_jaxnet_params(layers=[5,10,5]): 
    sender = layers[0]; 
    params = []
    for receiver in layers[1:]:
        weights = np.random.uniform(-.1,.1,(sender,receiver))  # Using numpy because this is not traced by jax
        biases = np.random.uniform(-.1,.1,(1,receiver))
        sender = receiver
        params.append([weights, biases])
    return params

def jaxnet_loss(params, features,targets):
    # Make a batched version of the `predict` function.
    # None => all params used on each call,
    # 0 => take one row at a time of the cases. vmap = vector map
    batched_predict = jax.vmap(predict, in_axes=(None, 0))
    predictions = batched_predict(params, features) 
    return jnp.mean(jnp.square(targets - predictions))

def sigmoid(x):
    """Activation function"""
    return 1 / (1 + jnp.exp(-x))

def predict(all_params, features):
    """Returned activations = the net's output"""
    activations = features
    for weights, biases in all_params:
        activations = sigmoid(jnp.dot(activations,weights) + biases) 
    
    return activations

def jaxnet_train_one_epoch(params,features,targets,lrate=0.1):
    # Mean squared error and gradient function with JAX tracing
    # Alternative: jax.grad(jaxnet_loss)(params, features,targets)
    mse, gradients = jax.value_and_grad(jaxnet_loss)(params, features,targets)
    return [(w - lrate * dw, b - lrate * db) for (w, b), (dw, db) in zip(params, gradients)], mse

def jaxnet_train(params,features,targets,epochs,lrate=0.1): 
    curr_params = params
    for _ in range(epochs):
        curr_params, mse = jaxnet_train_one_epoch(curr_params,features,targets,lrate)
        print(mse)
    return curr_params

def generate_data_cases(ncases=1000):
    features = np.random.uniform(-1,1,(ncases,5))
    targets = np.random.uniform(-1,1,(ncases,5))
    return features,targets

def jaxrun(epochs,ncases,layer_sizes,lrate=0.03): 
    features, targets = generate_data_cases(ncases)
    params = gen_jaxnet_params(layer_sizes) 
    jaxnet_train(params,features,targets,epochs,lrate=lrate)

if __name__ == "__main__":
    epochs = 100
    ncases = 1000
    layer_sizes = [5,10,5]
    lrate = 0.03
    jaxrun(epochs,ncases,layer_sizes,lrate=lrate)

