from getMoS2epsnk import *
import numpy as np    
import matplotlib.pyplot as plt
#
lambdas = np.linspace(400e-9,1000e-9,100)
Ef = 0.0
Temp = 300     
eps = getMoS2epsnk(lambdas,Ef,Temp)    

plt.figure(1)
plt.plot(lambdas*1e9,np.real(eps),label='Real')
plt.plot(lambdas*1e9,np.imag(eps),label='Imag')
plt.xlabel('Wavelength (nm)');
plt.ylabel('Permittivity');
plt.legend();    
plt.show()
    
n,k = eps2nk(eps)
plt.figure(2)
plt.plot(lambdas*1e9,n,label='n')
plt.plot(lambdas*1e9,k,label='k')
plt.xlabel('Wavelength (nm)');
plt.ylabel('Refractive Index');
plt.legend();    
plt.show()
