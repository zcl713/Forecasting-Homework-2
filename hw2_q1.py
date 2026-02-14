import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial
from scipy.stats import poisson

#Question 1a: Normal distribution
def normal_plot(mu, var, limit):
    x_values=np.linspace(-limit, limit, 500)
    y_values = (1/np.sqrt(2*np.pi*var))*np.exp((-(x_values-mu)**2)/(2*var))
    return x_values, y_values

x1,y1 = normal_plot(mu=0, var=1, limit=8)
x2,y2 = normal_plot(mu=-5, var=6, limit=15)
x3,y3 = normal_plot(mu=100, var=0.5, limit=105)

plt.figure()
plt.plot(x1,y1)
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title("Normal curve with mean=0 and variance=1")

plt.figure()
plt.plot(x2,y2)
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title("Normal curve with mean=-5 and variance=6")

plt.figure()
plt.plot(x3,y3)
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title("Normal curve with mean=100 and variance=0.5")

# Question 1b: Poisson distribution
def poisson_plot(l, limit):
    x_values=np.arange(0, limit)
    y_values = poisson.pmf(x_values, l)
    #y_values = np.exp(-l)*(l**x_values)/factorial(x_values)
    return x_values, y_values

x4,y4 = poisson_plot(l=0.03, limit=10)
x5,y5 = poisson_plot(l=50, limit=80)
x6,y6 = poisson_plot(l=1000, limit=1100)

plt.figure()
plt.stem(x4,y4, basefmt=' ')
plt.xlabel('x')
plt.ylabel('Probability')
plt.title("Poisson curve with lambda=0.03")

plt.figure()
plt.stem(x5,y5, basefmt=' ')
plt.xlabel('x')
plt.ylabel('Probability')
plt.title("Poisson curve with lambda=50")

plt.figure()
plt.stem(x6,y6, basefmt=' ')
plt.xlabel('x')
plt.ylabel('Probability')
plt.title("Poisson curve with lambda=1000")

plt.show()

