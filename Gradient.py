from sklearn.metrics import mean_squared_error
import numpy as np

# calculate using target function
def gradient_loss(w_past,x_train,y_train):
    
    n = x_train.shape[1]
    total_grad = np.zeros((1,n))
    loss = 0
    
    for i in range(x_train.shape[0]-1):
        x = x_train[i].reshape((-1,1)) # 1, x, x^2, x^3, x^4, x^5
        y = y_train[i]
        error = np.matmul(w_past, x) - y
        error = error.item() # calculate error
        loss = loss + error**2 # loss function
        grad = 2*error*np.transpose(x)
        total_grad = total_grad + grad
        
    return total_grad, loss
        
def train_set(X,Y,N,rng):
    sample_indices = np.arange(X.shape[0])
    indices = rng.choice(sample_indices,
                                   size=N,
                                   replace=False)
    return X[indices], Y[indices]

def stochastic_gradient(x_train, Y_train, lr=0.0001, epoch=5000, batch_size=128):
    
    
    # create random vector 6x1
    w = np.random.rand(x_train.shape[1])
    w = w.reshape((1,-1))

    loss_history = []
    random_seed = 1
    rng = np.random.RandomState(random_seed)
    
    for i in range(epoch):
        
        train_x, train_y = train_set(x_train, Y_train, batch_size, rng)
        grad, loss = gradient_loss(w,train_x, train_y)
        
        rmse = np.sqrt(1/batch_size * loss)
        loss_history.append(rmse)
        
       
        if rmse < 0.5:
            break
    
        w = w - lr * grad
        
    return loss_history, w[0]

def momentum_gradient(x_train, Y_train, lr=0.0001, epoch=5000, batch_size=128,gamma=0.5):
    # create random vector 6x1
    w = np.random.rand(6)
    w = w.reshape((1,6))

    v = 0 * w


    # update weight
 

    loss_history_momentum = []
    random_seed = 1
    rng = np.random.RandomState(random_seed)

    for i in range(epoch):
    

        train_x, train_y = train_set(x_train, Y_train, batch_size, rng)
    
        grad, loss = gradient_loss(w,train_x, train_y)
    
        loss_history_momentum.append(loss)
        if loss < 1:
            break
        v = gamma*v +lr*grad
        w = w - v
    return loss_history_momentum, w


def nestrov_gradient(x_train, Y_train, lr=0.0001, epoch=5000, batch_size=128,gamma=0.5):
    # create random vector 6x1
    w = np.random.rand(6)
    w = w.reshape((1,6))

    v = 0 * w


    # update weight
    

    loss_history_nag = []
    random_seed = 1
    rng = np.random.RandomState(random_seed)

    for i in range(epoch):
    

        train_x, train_y = train_set(x_train, Y_train, batch_size, rng)
    
        future_position = w - gamma*v
    
        grad, loss = gradient_loss(future_position,train_x, train_y)
    
        loss_history_nag.append(loss)
        if loss < 1:
            break
    
    
        v = gamma*v +lr*grad
        w = w - v
    return loss_history_nag, w

