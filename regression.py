import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

def with_i(x, coeffs):
    o = len(coeffs)
    y = 0
    for i in range(o):
        y += coeffs[i]*x**(i)
    return y

def with_coeff(x,y,n):
    left_matrix=[[0 for i in range(n+1)]*1 for i in range(n+1)]
    for i in range(n+1):
        for j in range(n+1):
            left_matrix[i][j]=sum([x[k]**(i+j) for k in range(len(x))])
    right_matrix=[[0] for i in range(n+1)]
    for k in range(n+1):
        right_matrix[k][0]=sum([y[i]*(x[i]**k) for i in range(len(x))])

    inv_matrix=np.linalg.inv(left_matrix)
    # print(inv_matrix)
    output_matrix=np.dot(inv_matrix,right_matrix)
    # print(output_matrix)
    # output_matrix = list(map(np.ndarray.tolist,output_matrix))

    return output_matrix

def regression_with_intercept(x,y,n):
    curv=with_coeff(x,y,n)
    curv_coff=[]
    for i in curv:
        curv_coff.append(round(i[0],3))
    plt.scatter(x, y, color= "green", marker= "*",label='data')
    y_poly = [with_i(i, curv) for i in x]
    plt.plot(x,y_poly,':',label=curv_coff)
    plt.title('Regression with Intercept , Order = '+str(n), fontweight="bold")
    plt.ylabel('y', fontweight='bold')
    plt.xlabel('x', fontweight='bold')
    plt.legend()
    plt.show()
    return curv


def without_i(x, coeffs):
    o = len(coeffs)
    y = 0
    for i in range(o):
        y += coeffs[i]*x**(i+1)
    return y

def without_coeff(x,y,n):
    left_matrix=[[0 for i in range(n)]*1 for i in range(n)]
    for i in range(n):
        for j in range(n):
            left_matrix[i][j]=sum([x[k]**(i+j+1+1) for k in range(len(x))])
    right_matrix=[[0] for i in range(n)]
    for k in range(n):
        right_matrix[k][0]=sum([y[i]*(x[i]**(k+1)) for i in range(len(x))])

    inv_matrix=np.linalg.inv(left_matrix)
    # print(inv_matrix)
    output_matrix=np.dot(inv_matrix,right_matrix)
    # print(output_matrix)
    # output_matrix = list(map(np.ndarray.tolist,output_matrix))
    return output_matrix

def regression_without_intercept(x,y,n):
    curv=without_coeff(x,y,n)
    curv_coff=[]
    for i in curv:
        curv_coff.append(round(i[0],3))
    plt.scatter(x, y, color= "green", marker= "*",label='data')
    y_poly = [without_i(i, curv) for i in x]
    plt.plot(x,y_poly,':',label=curv_coff)
    plt.title('Regression without Intercept , Order = '+str(n), fontweight="bold")
    plt.ylabel('y', fontweight='bold')
    plt.xlabel('x', fontweight='bold')
    plt.legend()
    plt.show()
    return curv


def logger(u):
    return np.log(u)

def power_model(x,y):
    x=list(map(logger,x))
    y=list(map(logger,y))
    resultt=with_coeff(x,y,n=1)
    resultt=list(map(np.ndarray.tolist,resultt))
    return [np.exp(resultt[0][0]),resultt[1][0]]

def power_eq(x,coeff):
        return coeff[0]*x**coeff[1]

def regression_power_model(x,y):
    curv=power_model(x,y)
    plt.scatter(x, y, color= "green", marker= "*",label='data')
    y_poly = [power_eq(i, curv) for i in x]
    for i in range(len(curv)):
        curv[i]=round(curv[i],2)
    plt.plot(x,y_poly,':',label=curv)
    plt.title('Regression of Power model $y=ax^b$', fontweight="bold")
    plt.ylabel('y', fontweight='bold')
    plt.xlabel('x', fontweight='bold')
    plt.legend()
    plt.show()
    return curv


def expo_model(x,y):
    y=list(map(logger,y))
    resultt=with_coeff(x,y,n=1)
    resultt=list(map(np.ndarray.tolist,resultt))
    return [np.exp(resultt[0][0]),resultt[1][0]]

def expo_eq(x,coeff):
    return coeff[0]*np.exp(x*coeff[1])

def regression_expo_model(x,y):
    curv=expo_model(x,y)
    plt.scatter(x, y, color= "green", marker= "*",label='data')
    y_poly = [expo_eq(i, curv) for i in x]
    for i in range(len(curv)):
        curv[i]=round(curv[i],2)
    plt.plot(x,y_poly,':',label=curv)
    plt.title('Regression of expo model $y=ae^{bx}$', fontweight="bold")
    plt.ylabel('y', fontweight='bold')
    plt.xlabel('x', fontweight='bold')
    plt.legend()
    plt.show()
    return curv




if __name__ =='__main__':

    x=[i+1 for i in range(10)]
    # y=[1,3,5,8,11,16,16,17,35,38]
    y=[5.5,7.7,9.5,11,12.2,13.4,14.5,15.5,16.5,17.3]
    y1=[8.2,14,21,35,62,100,140,250,450,740]

    order = 1

    coffecents = regression_with_intercept(x,y,order)
    coff = regression_without_intercept(x,y,order)
    pow_coef=regression_power_model(x,y)
    expo_coeff=regression_expo_model(x,y1)
