from math import floor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def createInputMatrix(n):
    X = []
    for i in range(0,n):
        X.append([1,i])
    return X

def linearRegression(dataFrame):
    """Finds a Linear Regression for a set of data from dataFrame input.

    Args:
        dataFrame (list): raw data input 

    Returns:
        (a0 , a1)(tuple) : a0 & a1 in equation {Y = a0 + a1.X}
    """
    DataNum = len(dataFrame)
    TrainT = np.array(createInputMatrix(floor(DataNum*0.95)))
    Y =  dataFrame.iloc[:318,1:].values
    
    
    
    
    
    print(DataNum)
    
    
    print(TrainT)
    print(dataFrame.iloc[:318,[1]].values)    
    # return (a0 , a1)

def quadraticRegression(dataFrame):
    """Finds a Quadratic Regression for a set of data from dataFrame input.

    Args:
        dataFrame (list): raw data input 

    Returns:
        (a0 , a1 , a2) (tuple) : a0 ,a1 , a2 in equation {Y = a0 + a1.X + a2.X^2}
    """
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
    
    product = np.matmul(ATA1, np.matmul(AT , Y))
    
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

def printGraph(X , Y):
    """Prints a graph by using matplotlib graph generator

    Args:
        X (list): Set of Xs
        Y (list): Set of Ys
    """
    pass
    
    

dataFrame = pd.read_csv("../covid_cases.csv")
linearRegression(dataFrame)