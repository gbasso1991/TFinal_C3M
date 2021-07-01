#%%
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from numpy.lib.nanfunctions import _remove_nan_1d

data = np.loadtxt('datos_tema_6.dat',skiprows=1)
tiempo = data[:,0] #[s]
temperatura = data[:,1]#[K]
campo = data[:,2]*(1000/4*pi)#[A/m]
mom_mag =  data[:,3]/1000#[Am^2]

fig,ax=plt.subplots()
ax.plot(campo[:len(campo)//2],mom_mag[:len(mom_mag)//2],'.-',lw=0.5)
ax.plot(campo[len(campo)//2:],mom_mag[len(mom_mag)//2:],'.-',lw=0.5)
ax.axhline(0,0,1,color='k',lw=0.2)
ax.axvline(0,0,1,color='k',lw=0.2)
#ax.axhline(max(mom_mag),0,1,color='k',lw=0.7)
#ax.axvline(campo[82],0,1,color='r',lw=0.7)
plt.xlabel('$H$ $(A/m)$')
plt.ylabel('$\mu$ $(Am^2)$')
plt.title('Coloide de NPM de Fe en hexano')
plt.xlim(-0.4e6,0.4e6)
#plt.ylim(-2.5e-6,2.5e-6)


###########################################
#%% Ajuste Bajo campo (LF) y a alto campo (HF)
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def ajuste_LF(x,a,b):
    return a*x+b
#%%
x_LF_1 = np.sort(campo[44:64])#1er rama
y_LF_1 = np.sort(mom_mag[44:64])

x_LF_2 = np.sort(campo[153:173])#2da rama
y_LF_2 = np.sort(mom_mag[153:173])
 
param_1,_ = curve_fit(ajuste_LF,x_LF_1,y_LF_1)
param_2,_ = curve_fit(ajuste_LF,x_LF_2,y_LF_2)

m1,n1=param_1
print('mu = {}*H + {}'.format(m1, n1))
m2,n2=param_2
print('mu = {}*H + {}'.format(m2, n2))
g_1 = ajuste_LF(x_LF_1,m1,n1)
g_2 = ajuste_LF(x_LF_2,m2,n2)
#%% Figura Ajuste LF 
plt.figure()
plt.plot(x_LF_1/1000,y_LF_1,'.-', label='1er rama')
plt.plot(x_LF_1/1000,g_1,label='LF1')
plt.plot(x_LF_2/1000,y_LF_2,'.-',label='2da rama')
plt.plot(x_LF_2/1000,g_2,label='LF2')
plt.grid()
plt.legend()
plt.title('Ajuste Lineal a bajo campo\n$H_m={:.2e}$ $kA/m$'.format(np.max(campo)/1000))
plt.ylabel('$\mu$ $(Am^2)$')
plt.xlabel('$H$ $(kA/m)$')
plt.show()

print('''Pendiente: 
        m1={:.4e} m^3
        m2={:.4e} m^3
Ordenada:
        n1={:.4e} Am^2
        n2={:.4e} Am^2
Calidad del ajuste:
        R1^2= {:.4f} 
        R2^2= {:.4f} 

'''.format(m1,m2,n1,n2,r2_score(y_LF_1,g_1),r2_score(y_LF_2,g_2)))

###################################

#%% Altos campos - Ajuste No lineal
def ajuste_HF(H,a,mu_s,C):
    return (1-a/H)*mu_s + C*H

x_HF_1 = np.sort(campo[0:30])#1er rama
y_HF_1 = np.sort(mom_mag[0:30])

x_HF_2 = np.sort(campo[188:218])#2da rama
y_HF_2 = np.sort(mom_mag[188:218])
 
#fig,ax=plt.subplots()
#ax.plot(x_HF_1,y_HF_1,'.-',lw=0.5)
#ax.plot(x_HF_2,y_HF_2,'.-',lw=0.5)
#ax.axhline(0,0,1,color='k',lw=0.2)
#ax.axvline(0,0,1,color='k',lw=0.2)
#ax.axhline(max(mom_mag),0,1,color='k',lw=0.7)
#ax.axvline(campo[82],0,1,color='r',lw=0.7)
##ax.grid()
#plt.xlabel('$H$ $(A/m)$')
#plt.ylabel('$\mu$ $(Am^2)$')
#plt.title('Coloide de NPM de Fe en hexano')
#plt.xlim(0,np.max(campo))
#plt.ylim(-2.5e-6,2.5e-6)
#plt.show()

parametros_1,_ = curve_fit(ajuste_HF,x_HF_1,y_HF_1)
parametros_2,_ = curve_fit(ajuste_HF,x_HF_2,y_HF_2)

a1,mu_s1,C1=parametros_1
a2,mu_s2,C2=parametros_2

h_1= ajuste_HF(x_HF_1,a1,mu_s1,C1)
h_2= ajuste_HF(x_HF_2,a2,mu_s2,C2)

plt.figure()
plt.plot(x_HF_1,y_HF_1,'.-', label='1er rama')
plt.plot(x_HF_1,h_1,label='HF1')
plt.plot(x_HF_2,y_HF_2,'.-',label='2da rama')
plt.plot(x_HF_2,h_2,label='HF2')
plt.legend()
plt.title('Ajuste no lineal para campos altos\n$\mu \simeq (1- a/H) \mu_s + CH$')
plt.show()

print(''' 
a1= {:.4e} m/A
a2= {:.4e} m/A

mu_s1= {:.4e} Am^2
mu_s2= {:.4e} Am^2

C1= {:.4e} m^3
C2= {:.4e} m^3

R1^2= {:.4f} 
R2^2= {:.4f}'''.format(a1,a2,mu_s1,mu_s2,C1,C2,r2_score(y_HF_1,h_1),r2_score(y_HF_2,h_2)))


#%% Ajusto en la rama de campos altos negativos

#%% seleccion del campo para cada ajuste
plt.plot(campo,'.-')
plt.axhline(campo[0]  ,0,0.2,color='r')
plt.axhline(campo[30] ,0,0.2,color='r')
plt.axhline(campo[-30],0.8,1,color='r')
plt.axhline(campo[-1],0.8,1,color='r')

plt.axhline(campo[78] ,0.2,0.45,color='g')
plt.axhline(campo[108],0.3,0.5,color='g')
plt.axhline(campo[108],0.5,0.7,color='g')
plt.axhline(campo[139],0.55,0.8,color='g')
plt.title('Campo Magnetico')
plt.grid()


#%% Rama negativa
x_HF_3 = np.sort(campo[78:108])
y_HF_3 = np.sort(mom_mag[78:108])

x_HF_4 = np.sort(campo[109:139])
y_HF_4 = np.sort(mom_mag[109:139])
 
parametros_3,_ = curve_fit(ajuste_HF,x_HF_3,y_HF_3)
parametros_4,_ = curve_fit(ajuste_HF,x_HF_4,y_HF_4)

a3,mu_s3,C3=parametros_3
a4,mu_s4,C4=parametros_4

h_3 = ajuste_HF(x_HF_3,a3,mu_s3,C3)
h_4 = ajuste_HF(x_HF_4,a4,mu_s4,C4)
plt.figure()
#plt.plot(x_HF_3,y_HF_3,'.-', label='1er rama')
plt.plot(x_HF_3,h_3,label='HF3')
#plt.plot(x_HF_4,y_HF_4,'.-',label='2da rama')
plt.plot(x_HF_4,h_4,label='HF4')
plt.legend()
plt.title('Ajuste No Lineal para campos altos\n$\mu \simeq (1- a/H) \mu_s + CH$')
plt.show()
print('''
a3= {:.4e} m/A
a4= {:.4e} m/A

mu_s3= {:.4e} Am^4
mu_s4= {:.4e} Am^4

C3= {:.4e} m^3
C4= {:.4e} m^3

R3^4= {:.4f} 
R4^4= {:.4f}'''.format(a3,a4,mu_s3,mu_s4,C3,C4,r2_score(y_HF_3,h_3),r2_score(y_HF_4,h_4)))


#%%

