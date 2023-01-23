import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, jit, nn
import time

# Specify paths
repo_path = '/Users/sumiya/git/LastBlackBox'
box_path = repo_path + '/boxes/learning'
data_path = box_path + '/supervised/_data/complex.csv'

# Load data
data = np.genfromtxt(data_path, delimiter=',')
x = data[:,0]
y = data[:,1]
x = np.expand_dims(x,1) # Add dimension
y = np.expand_dims(y,1) # Add dimension

def main():
    start_time = time.time()
    # Define network (size of hidden layer)
    num_hidden_neurons = 10

    # Initalize hidden layers
    W1 = np.random.rand(num_hidden_neurons) - 0.5
    B1 = np.random.rand(num_hidden_neurons) - 0.5
    W1 = np.expand_dims(W1,0)
    B1 = np.expand_dims(B1,0)

    W2 = np.random.rand(num_hidden_neurons, num_hidden_neurons) - 0.5
    B2 = np.random.rand(num_hidden_neurons, num_hidden_neurons) - 0.5
    # B2 = np.random.rand(num_hidden_neurons) - 0.5
    # B2 = np.expand_dims(B2,0)

    W3 = np.random.rand(num_hidden_neurons, num_hidden_neurons) - 0.5
    B3 = np.random.rand(num_hidden_neurons, num_hidden_neurons) - 0.5
    # B3 = np.random.rand(num_hidden_neurons) - 0.5
    # B3 = np.expand_dims(B3,0)

    W4 = np.random.rand(num_hidden_neurons) - 0.5
    B4 = np.random.rand(num_hidden_neurons) - 0.5
    W4 = np.expand_dims(W4,0)
    B4 = np.expand_dims(B4,0)

    # Define function (network)
    def func(x, W1, B1, W2, B2, W3, B3, W4, B4):
        all_one = jnp.ones((len(x), num_hidden_neurons))
        hidden_1 = x.dot(W1) + B1
        activations_1 = nn.sigmoid(hidden_1)
        hidden_2 = activations_1.dot(W2) + all_one.dot(B2)
        # hidden_2 = activations_1.dot(W2) + B2
        activations_2 = nn.sigmoid(hidden_2)
        hidden_3 = activations_2.dot(W3) + all_one.dot(B3)
        # hidden_3 = activations_2.dot(W3) + B3
        activations_3 = nn.sigmoid(hidden_3)
        interim = activations_3.dot(W4.T) + B4
        output = jnp.sum(interim, axis=1)
        return output

    # Define loss (mean squared error)
    def loss(x, y, W1, B1, W2, B2, W3, B3, W4, B4):
        guess = func(x, W1, B1, W2, B2, W3, B3, W4, B4)
        err = np.squeeze(y) - guess
        return jnp.mean(err*err)

    # Compute gradient (w.r.t. parameters)
    grad_loss_W1 = jit(grad(loss, argnums=2))
    grad_loss_B1 = jit(grad(loss, argnums=3))
    grad_loss_W2 = jit(grad(loss, argnums=4))
    grad_loss_B2 = jit(grad(loss, argnums=5))
    grad_loss_W3 = jit(grad(loss, argnums=6))
    grad_loss_B3 = jit(grad(loss, argnums=7))
    grad_loss_W4 = jit(grad(loss, argnums=8))
    grad_loss_B4 = jit(grad(loss, argnums=9))

    # Train
    alpha = .0001
    beta = 0.99
    deltas_W1 = np.zeros(num_hidden_neurons)
    deltas_B1 = np.zeros(num_hidden_neurons)
    deltas_W2 = np.zeros(num_hidden_neurons)
    deltas_B2 = np.zeros(num_hidden_neurons)
    deltas_W3 = np.zeros(num_hidden_neurons)
    deltas_B3 = np.zeros(num_hidden_neurons)
    deltas_W4 = np.zeros(num_hidden_neurons)
    deltas_B4 = np.zeros(num_hidden_neurons)
    report_interval = 100
    num_steps = 10000
    initial_loss = loss(x, y, W1, B1, W2, B2, W3, B3, W4, B4)
    losses = [initial_loss]
    for i in range(num_steps):    

        # Compute gradients
        gradients_W1 = grad_loss_W1(x, y, W1, B1, W2, B2, W3, B3, W4, B4)
        gradients_B1 = grad_loss_B1(x, y, W1, B1, W2, B2, W3, B3, W4, B4)
        gradients_W2 = grad_loss_W2(x, y, W1, B1, W2, B2, W3, B3, W4, B4)
        gradients_B2 = grad_loss_B2(x, y, W1, B1, W2, B2, W3, B3, W4, B4)
        gradients_W3 = grad_loss_W3(x, y, W1, B1, W2, B2, W3, B3, W4, B4)
        gradients_B3 = grad_loss_B3(x, y, W1, B1, W2, B2, W3, B3, W4, B4)
        gradients_W4 = grad_loss_W4(x, y, W1, B1, W2, B2, W3, B3, W4, B4)
        gradients_B4 = grad_loss_B4(x, y, W1, B1, W2, B2, W3, B3, W4, B4)

        # Update deltas
        deltas_W1 = (alpha * gradients_W1) + (beta * deltas_W1)
        deltas_B1 = (alpha * gradients_B1) + (beta * deltas_B1)
        deltas_W2 = (alpha * gradients_W2) + (beta * deltas_W2)
        deltas_B2 = (alpha * gradients_B2) + (beta * deltas_B2)
        deltas_W3 = (alpha * gradients_W3) + (beta * deltas_W3)
        deltas_B3 = (alpha * gradients_B3) + (beta * deltas_B3)
        deltas_W4 = (alpha * gradients_W4) + (beta * deltas_W4)
        deltas_B4 = (alpha * gradients_B4) + (beta * deltas_B4)

        # Update parameters
        W1 -= (deltas_W1)
        B1 -= (deltas_B1)
        W2 -= (deltas_W2)
        B2 -= (deltas_B2)
        W3 -= (deltas_W3)
        B3 -= (deltas_B3)
        W4 -= (deltas_W4)
        B4 -= (deltas_B4)

        # Store loss
        final_loss = loss(x, y, W1, B1, W2, B2, W3, B3, W4, B4)
        losses.append(final_loss)

        # Report?
        if((i % report_interval) == 0):
            np.set_printoptions(precision=3)
            print("MSE: {0:.2f}".format(final_loss))

    # Compare prediction to data
    prediction = func(x, W1, B1, W2, B2, W3, B3, W4, B4)
    print("--- %s seconds ---" % (time.time() - start_time))

    plt.subplot(1,2,1)
    plt.plot(x, y, 'b.', markersize=1)              # Plot data
    plt.plot(x, prediction, 'r.', markersize=1)     # Plot prediction
    plt.subplot(1,2,2)
    plt.plot(np.array(losses))                      # Plot loss over training
    plt.show()

if __name__ == "__main__":
    main()