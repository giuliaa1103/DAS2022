# 26 Apr 2022
#

from tabnanny import verbose
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from matplotlib import pyplot as plt

plt.close('all')
mndata = MNIST('python-mnist/data')

np.random.seed(10)

###############################################################################

T = 3  # Layers
d = 784  # Number of neurons in each layer. Same numbers for all the layers
N_images = 50
N_class = 10
# Training Set
#label_point = np.random.randn(d) # D = x0
#data_point = 3*np.random.uniform(-1,1,size=d) # y = xT = phi (ustar)
images,labels = mndata.load_training()


# Gradient Method Parameters
max_iters = 4 # epochs
stepsize = 0.1 # learning rate
print("the first label is ", labels[0])
###############################################################################
# Creation of the one hot encoded vector y

def one_hot_encoded(label):
  vect_y = np.zeros((N_class,), dtype=int)
  vect_y[label] = 1
  return vect_y 

print(one_hot_encoded(labels[0]))
# Activation Function
def sigmoid_fn(xi):
  return 1/(1+np.exp(-xi))

# Derivative of Activation Function
def sigmoid_fn_derivative(xi):
  return sigmoid_fn(xi)*(1-sigmoid_fn(xi))


# Inference: x_tp = f(xt,ut)
def inference_dynamics(xt,ut):
  """
    input: 
              xt current state
              ut current input
    output: 
              xtp next state
  """
  xtp = np.zeros(d)
  for ell in range(d):
    temp = xt@ut[ell,1:] + ut[ell,0] # including the bias

    xtp[ell] = sigmoid_fn( temp ) # x' * u_ell
  
  return xtp


# Inference: x_tp = f(xt,ut)
def inference_dynamics_FINALSTEP(xTminusOne,uT):
  """
    input: 
              xt current state
              ut current input
    output: 
              xtp next state
  """
  xtp = np.zeros(N_class)
  for ell in range(N_class):
    temp = uT[ell,1:] @ xTminusOne + uT[ell,0] # including the bias

    xtp[ell] = sigmoid_fn( temp ) # x' * u_ell
  
  return xtp


# Forward Propagation
def forward_pass(uu,x0,uu_T):
  """
    input: 
              uu input trajectory: u[0],u[1],..., u[T-1]
              uu_T input at final layer: u[T]
              x0 initial condition
    output: 
              xx state trajectory: x[1],x[2],..., x[T]
  """
  xx = np.zeros((T,d))
  xx_T = np.zeros((N_class))
  xx[0] = x0

  for t  in range(T-1):
    xx[t+1] = inference_dynamics(xx[t],uu[t]) # x^+ = f(x,u)
  xx_T= inference_dynamics_FINALSTEP(xx[T-1],uu_T)

  return xx,xx_T


# Adjoint dynamics: 
#   state:    lambda_t = A.T lambda_tp
#   output: deltau_t = B.T lambda_tp
def adjoint_dynamics_LASTSTEP(lTplusOne,xT,uT):
  """
    input: 
              lTplusOne gradient of J
              xT 784x1 last but one layer
              uT parameters in the last layer
    output: 
              lt previous costate
              delta_ut loss gradient wrt u_t
  """
  df_dx = np.zeros((d,N_class))#A_Transpose 784x10

  # df_du = np.zeros((d,(d+1)*d))
  Delta_uT = np.zeros((N_class,d+1))#It is 10x(d+1), in a layer there are a number of parameters equal to the number of neurons times number of parameters in each neurons

  for j in range(N_class):
    dsigma_j = sigmoid_fn_derivative(uT[j,1:]@xT + uT[j,0]) 

    df_dx[:,j] =uT[j,1:]*dsigma_j
    # df_du[j, XX] = dsigma_j*np.hstack([1,xt])
    
    # B'@ltp
    Delta_uT[j,0] = lTplusOne[j]*dsigma_j
    Delta_uT[j,1:] = xT*lTplusOne[j]*dsigma_j
  
  lt = df_dx@lTplusOne # A'@ltp
  # Delta_ut = df_du@ltp

  return lt, Delta_uT

# Adjoint dynamics: 
#   state:    lambda_t = A.T lambda_tp
#   output: deltau_t = B.T lambda_tp
def adjoint_dynamics(ltp,xt,ut):
  """
    input: 
              llambda_tp current costate
              xt current state
              ut current input
    output: 
              llambda_t next costate
              delta_ut loss gradient wrt u_t
  """
  df_dx = np.zeros((d,d))

  # df_du = np.zeros((d,(d+1)*d))
  Delta_ut = np.zeros((d,d+1))

  for j in range(d):
    dsigma_j = sigmoid_fn_derivative(xt@ut[j,1:] + ut[j,0]) 

    df_dx[:,j] =ut[j,1:]*dsigma_j
    # df_du[j, XX] = dsigma_j*np.hstack([1,xt])
    
    # B'@ltp
    Delta_ut[j,0] = ltp[j]*dsigma_j
    Delta_ut[j,1:] = xt*ltp[j]*dsigma_j
  
  lt = df_dx@ltp # A'@ltp
  # Delta_ut = df_du@ltp

  return lt, Delta_ut

# Backward Propagation
def backward_pass(xx,uu,llambdaTplusOne,uu_T):
  """
    input: 
              xx state trajectory: x[1],x[2],..., x[T]
              uu input trajectory: u[0],u[1],..., u[T-1]
              llambdaT terminal condition
    output: 
              llambda costate trajectory
              delta_u parameters update  (T-1)*d*(d+1)
              Delta_u_T parameters update of the last layer  N_class*(d+1)
  """
  

  Delta_u = np.zeros((T-1,d,d+1))
  Delta_u_T = np.zeros((N_class,d+1))

  llambdaT,Delta_u_T = adjoint_dynamics_LASTSTEP(llambdaTplusOne,xx[T-1],uu_T) #T-1

  llambda = np.zeros((T,d))
  llambda[-1] = llambdaT
  for t in reversed(range(T-1)): # T-2,T-1,...,1,0
    llambda[t], Delta_u[t] = adjoint_dynamics(llambda[t+1],xx[t],uu[t])
  

  return Delta_u,Delta_u_T

def verification(p_score,label_i):
  """
    input: 
              p_score - vector 10x1 containing the probabilities
              label[i] - scalar 
              
    output: 
              binaryResult - True or False 
  """ 
  binaryResult = np.argmax(p_score) == label_i
  verbose=0
  if verbose:
    print('the score is {0} and the label is {1}. So the result will be {2}'.format(p_score,label_i,binaryResult))
  
  return binaryResult

def accuracy(images,num_of_images, labels):
  """
    input: 
              images - vector of images 
              uu - parameters of the main network
              uu_T - parameters of the final layer
               
              
    output: 
              accuracy_value - a number belonging to (0,1) 
  """ 
  count = 0
  for i in range(num_of_images):
    
    _,xx_T = forward_pass(uu,images[i],uu_T) # T x d
    #print('I am evaluating the ', i , ' image')
    _,p_score = soft_max(xx_T)
    count = count + int(verification(p_score,labels[i]))
  
  accuracy_value = count/num_of_images

  return accuracy_value
    
  


# Linear classifier function
def soft_max(xx_T):
  """
    input: 
              xx_T final state trajectory: x[T]
    output: 
              score [100, 0, 154, ... 0]
              probability score [0.2, 0, 0.8, ... 0]
  """ 
  score = xx_T
  p_score = np.zeros(N_class)
  sum_temp = sum(np.exp(score))
  for i in range (N_class):
    p_score[i] = np.exp(score[i])/sum_temp
  return score,p_score


###############################################################################
# MAIN
###############################################################################

J = np.zeros((max_iters))                       # Cost

# Initial Weights / Initial Input Trajectory
uu = np.random.randn(T-1, d, d+1)
uu_T = np.random.randn(N_class, d+1)

# Initial State Trajectory
### xx,xx_T = forward_pass(uu,images[0],uu_T) # T x d

Delta_u_list = np.zeros((N_images,T-1,d,d+1))
Delta_u_T_list = np.zeros((N_images,N_class,d+1))
# scores, array 10x1 for each images

Delta_u = np.zeros((T-1, d, d+1))
Delta_u_T = np.zeros((N_class, d+1))

# GO!
for k in range(max_iters):
  if k%2 == 0:
    print('Cost at k={:d} is {:.4f}'.format(k,J[k-1]))
  
  for i in range(N_images):
    
    xx,xx_T = forward_pass(uu,images[i],uu_T) # T x d
    #print('I am evaluating the ', i , ' image')
    score,p_score = soft_max(xx_T)

    result =verification(p_score,labels[i])

    # Backward propagation
    llambdaTplusOne = 2*( p_score - one_hot_encoded(labels[i])) # xT
    Delta_u,Delta_u_T  = backward_pass(xx,uu,llambdaTplusOne,uu_T) # the gradient of the loss function 
    Delta_u_list[i]= Delta_u
    Delta_u_T_list[i] = Delta_u_T
    # Store the Loss Value across Iterations
    J[k] = J[k] + (p_score - one_hot_encoded(labels[i]))@(p_score - one_hot_encoded(labels[i])) # it is the cost at k+1
    # np.linalg.norm( xx[-1,:] - label_point )**2

    

  # Update the weights
  uu = uu - stepsize*sum(Delta_u_list) # overwriting the old value
  uu_T = uu_T - stepsize*sum(Delta_u_T_list)

print(accuracy(images[300:400],100,labels[300:400]))


print(xx[-1,:].shape)
_,ax = plt.subplots()
ax.plot(range(max_iters),J)
plt.show()
#plt.imshow(np.array(images[10]).reshape(28,28),cmap='gray')
#plt.show()


