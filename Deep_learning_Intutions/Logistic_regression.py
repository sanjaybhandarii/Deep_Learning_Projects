import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os


def sigmoid(x):
    s= 1/(1+np.exp(-x))
    return s



def initialize_zeros(dim):
    w = np.zeros((dim,1))
    b = 0

    return w, b

def propagate(w,b,X,Y):
    m = len(X)
    A = sigmoid(np.dot(w.T,X)+b)
    cost = np.sum((-Y*np.log(A)-(1-Y)*np.log(1-A)))/m

    dw = np.dot(X,(A-Y).T)/m
    db = np.sum(A-Y)/m
  

    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


def  optimize(w,b,X,Y,num_iterations,learning_rate, print_cost =False):
    
    costs =[]

    for i in range(num_iterations):
        grads, cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - (learning_rate*dw)
        b = b - (learning_rate*db)

        if i%100 == 0:
            costs.append(cost)

        if print_cost and i%100 ==0:
            print ("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

# GRADED FUNCTION: optimize

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
  
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ### 
        grads, cost = propagate(w, b, X, Y)
        ### END CODE HERE ###
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - (learning_rate*dw)
        b = b - (learning_rate*db)
        ### END CODE HERE ###
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

def predict(w,b,X):


    m = len(X)
    Y_prediction = np.zeros((1,m))
    w = w.reshape(len(X), 1)    

    y_hat = sigmoid(np.dot(w.T,X)+b)

    Y_prediction = (y_hat >= 0.5) * 1.0

    return Y_prediction

#now bulid the logistic regression model
def model(X_train, Y_train,  num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w,b = initialize_zeros(len(X_train))
    parameters, grads, costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    w = parameters["w"]
    b = parameters["b"]

    # Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    # print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
        #  "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d




df = pd.read_csv('/home/chaos/Documents/GitHub/Deep_Learning_Projects/Python_progs/Deep_learning_Intutions/classification.csv')

X_train = df['age']
Y_train = df['success']



d = model(X_train, Y_train,  num_iterations = 4000, learning_rate = 0.001, print_cost = False)

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()



