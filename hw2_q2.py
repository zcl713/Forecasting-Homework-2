# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:40:28 2022

@author: waseem
"""

import numpy as np
import matplotlib.pyplot as plt  # To visualize

# Part a: Poisson Random Process with lambda=50 
### Stochastic process
t = np.arange(0, 100)
s1 = np.random.poisson(50,100)
s2 = np.random.poisson(50,100)
s3 = np.random.poisson(50,100)
s4 = np.random.poisson(50,100)

fig = plt.figure(figsize=(20, 14))
plt.suptitle('Poisson Random Process (λ = 50)', fontsize=24)

ax1 = plt.subplot(411)
plt.plot(t, s1)
plt.xlim(0, 100)
plt.ylim(30, 70)
plt.ylabel('$X_1$(t)',fontsize=20)
plt.tick_params('x', labelbottom=False)
plt.yticks(fontsize=12)
ax1.xaxis.grid(linewidth=2)
ax1.yaxis.grid(linewidth=1)

# share x only
ax2 = plt.subplot(412, sharex=ax1)
plt.plot(t, s2)
plt.xlim(0, 100)
plt.ylim(30, 70)
plt.ylabel('$X_2$(t)',fontsize=20)
# make these tick labels invisible
plt.tick_params('x', labelbottom=False)
plt.yticks(fontsize=12)
ax2.xaxis.grid(linewidth=2)
ax2.yaxis.grid(linewidth=1)

# share x only
ax3 = plt.subplot(413, sharex=ax1)
plt.plot(t, s3)
plt.xlim(0, 100)
plt.ylim(30, 70)
plt.ylabel('$X_3$(t)',fontsize=20)
# make these tick labels invisible
plt.tick_params('x', labelbottom=False)
plt.yticks(fontsize=12)
ax3.xaxis.grid(linewidth=2)
ax3.yaxis.grid(linewidth=1)

# share x and y
ax4 = plt.subplot(414, sharex=ax1, sharey=ax1)
plt.plot(t, s4)
plt.xlim(0, 100)
plt.ylim(30, 70)
plt.ylabel('$X_4$(t)',fontsize=20)
ax4.xaxis.grid(linewidth=2)
ax4.yaxis.grid(linewidth=1)
plt.xticks(fontsize=20)
# plt.savefig('Stochastic_Processes.png')
plt.show()

# Part b: Normal Random Process with mean=0 and variance=1
### Stochastic process
t = np.arange(0, 100)
s5 = np.random.normal(0,1,100)
s6 = np.random.normal(0,1,100)
s7 = np.random.normal(0,1,100)
s8 = np.random.normal(0,1,100)

fig = plt.figure(figsize=(20, 14))
plt.suptitle('Normal Random Process (μ = 0, σ² = 1)', fontsize=24)

ax5 = plt.subplot(411)
plt.plot(t, s5)
plt.xlim(0, 100)
plt.ylim(-5, 5)
plt.ylabel('$X_1$(t)',fontsize=20)
plt.tick_params('x', labelbottom=False)
plt.yticks(fontsize=12)
ax5.xaxis.grid(linewidth=2)
ax5.yaxis.grid(linewidth=1)

# share x only
ax6 = plt.subplot(412, sharex=ax5)
plt.plot(t, s6)
plt.xlim(0, 100)
plt.ylim(-5, 5)
plt.ylabel('$X_2$(t)',fontsize=20)
# make these tick labels invisible
plt.tick_params('x', labelbottom=False)
plt.yticks(fontsize=12)
ax6.xaxis.grid(linewidth=2)
ax6.yaxis.grid(linewidth=1)

# share x only
ax7 = plt.subplot(413, sharex=ax5)
plt.plot(t, s7)
plt.xlim(0, 100)
plt.ylim(-5, 5)
plt.ylabel('$X_3$(t)',fontsize=20)
# make these tick labels invisible
plt.tick_params('x', labelbottom=False)
plt.yticks(fontsize=12)
ax7.xaxis.grid(linewidth=2)
ax7.yaxis.grid(linewidth=1)

# share x and y
ax8 = plt.subplot(414, sharex=ax5, sharey=ax5)
plt.plot(t, s8)
plt.xlim(0, 101)
plt.ylim(-5, 5)
plt.ylabel('$X_4$(t)',fontsize=20)
ax8.xaxis.grid(linewidth=2)
ax8.yaxis.grid(linewidth=1)
plt.xticks(fontsize=20)
# plt.savefig('Stochastic_Processes.png')
plt.show()
