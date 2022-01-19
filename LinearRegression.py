from math import floor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def createInputMatrix(n, linear):
    
    if linear:
        X = np.zeros((n,2))
        for i in range(0,n):
            X[i][0] = 1
            X[i][1] = i
        return X
    else:
        X = np.zeros((n,3))
        for i in range(0,n):
            X[i][0] = 1
            X[i][1] = i
            X[i][2] = i**2
        return X


def linearRegression(TrainData , Y):
    """Finds a Linear Regression for a set of data from dataFrame input.

    Args:
        TrainData (list): raw data input of Xs
        Y(list): raw data input of Ys

    Returns:
        (a0 , a1)(tuple) : a0 & a1 in equation {Y = a0 + a1.X}
    """
    
    
    #Train X Input Matrix Transpose -> Train Transpose
    TT= TrainData.T
    
    #Transpose of Xs multiplied by Xs
    XTX = np.matmul(TT,TrainData)
    
    #Inverse of XTX
    XTX1 = Inverse(XTX)
    
    A = matrixChainMultiplication(Y , TT , XTX1)
        
    # test
    # print(TrainData)
    # print(TT)
    # print(XTX)
    # print(XTX1)
    # print(Y)   
    # print(A)
    
    a0 = A[0,0]
    a1 = A[1,0]
    print(a0 , a1) 
    return (a0 , a1)

def quadraticRegression(TrainData, Y):
    """Finds a Quadratic Regression for a set of data from dataFrame input.

    Args:
        TrainData (list): raw data input of Xs
        Y(list): raw data input of Ys
    Returns:
        (a0 , a1 , a2) (tuple) : a0 ,a1 , a2 in equation {Y = a0 + a1.X + a2.X^2}
    """
    
    #Train X Input Matrix Transpose -> Train Transpose
    TT= TrainData.T
    
    #Transpose of Xs multiplied by Xs
    XTX = np.matmul(TT,TrainData)
    
    #Inverse of XTX
    XTX1 = Inverse(XTX)
    
    A = matrixChainMultiplication(Y , TT , XTX1)
        
    # test
    # print(TrainData)
    # print(TT)
    # print(XTX)
    # print(XTX1)
    # print(Y)   
    # print(A)
    
    a0 = A[0,0]
    a1 = A[1,0]
    a2 = A[2,0]
    print(a0 , a1 , a2) 
    return (a0 , a1 , a2)


def Inverse(matrix):
    """Inverse Matrix (M^-1) Generator

    Args:
        matrix (nparray): input matrix

    Returns:
        nparray: Inverted matrix
    """
    
    inverse = np.linalg.inv(matrix)
    
    return inverse


def matrixChainMultiplication( Y , AT , ATA1):
    """Multiply a chain of matrices in optimal order

    Args:
        Y(nparray) : Ys matrix
        AT(nparray) : A Transpose
        ATA1(nparray) : inverse of A * AT
    Returns:
        nparray: product matrix
    """
    
    product = np.matmul(np.matmul(ATA1 , AT), Y)
    
    return product


def linearPredictaion(x , a0 , a1):
    """Predicts the Expected output by using the coefficients calculated by linearRegression.

    Args:
        x (double): x input
        a0 (double): a0 of the equation {Y= a0 + a1.X}
        a1 (double): a1 of the equation {Y= a0 + a1.X}
    
    Returns:
        Y(double): answer calculated by linearRegression prediction
    """
    return a0 + (a1*x)

def quadraticPrediction(x , a0 , a1 , a2):
    """Predicts the Expected output by using the coefficients calculated by quadraticRegression.

    Args:
        x (double): x input
        a0 (double): a0 of the equation {Y= a0 + a1.X+a2X^2}
        a1 (double): a1 of the equation {Y= a0 + a1.X+a2X^2}
        a2 (double): a2 of the equation {Y= a0 + a1.X+a2X^2}
    
    Returns:
        Y(double): answer calculated by quadraticRegression prediction
    """
    return a0 + (a1*x) + (a2 * pow(x, 2))

def printGraph(X , Y , lbl = ""):
    """Prints a graph by using matplotlib graph generator

    Args:
        X (list): Set of Xs
        Y (list): Set of Ys
    """
    plt.plot(X , Y , label= lbl)
    # plt.show()
    pass
    
    

dataFrame = pd.read_csv("../covid_cases.csv")
DataNum = len(dataFrame)
TrainData = np.array(createInputMatrix(floor(DataNum), True))
TrainDataQ = np.array(createInputMatrix(floor(DataNum), False))
Y =  dataFrame.iloc[:,[1]].values
Al = linearRegression(TrainData , Y)
Aq = quadraticRegression(TrainDataQ , Y)
X = TrainData[:,[1]]
printGraph(X , Al[0]+(Al[1]*X) , lbl= "Linear Regression:" + str(Al[0]) + "+"+ str(Al[1]) + "x")
printGraph(X , Aq[0]+(Aq[1]* X)+(Aq[2]*pow(X,2)) , lbl= "Quadratic Regression:" + str(Aq[0]) + " + "+ str(Aq[1]) + "x +" + str(Aq[2]) + "x^2" )
plt.scatter(X,Y , label ="raw data")
plt.legend()
plt.show()
