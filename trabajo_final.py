#%%
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from numpy.lib.nanfunctions import _remove_nan_1d
#%%
data = np.loadtxt('datos_tema_6.dat',skiprows=1)
tiempo = data[:,0] #[s]
temperatura = data[:,1]#[K]
campo = data[:,2]*(1000/4*pi)#[A/m]
mom_mag =  data[:,3]/1000#[Am^2]
#%%Ciclo
fig,ax=plt.subplots()
ax.plot(campo,mom_mag,'.-',lw=0.5)
#ax.plot(campo[:len(campo)//2],mom_mag[:len(mom_mag)//2],'.-',lw=0.5)
#ax.plot(campo[len(campo)//2:],mom_mag[len(mom_mag)//2:],'.-',lw=0.5)
ax.axhline(0,0,1,color='k',lw=0.2)
ax.axvline(0,0,1,color='k',lw=0.2)
#ax.axhline(max(mom_mag),0,1,color='k',lw=0.7)
#ax.axvline(campo[82],0,1,color='r',lw=0.7)
plt.xlabel('$H$ $(A/m)$')
plt.ylabel('$\mu$ $(Am^2)$')
plt.title('Coloide de NPM de Fe en hexano')
plt.grid()
#plt.xlim(-0.4e6,0.4e6)
#plt.ylim(-2.5e-6,2.5e-6)


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
plt.xlabel('Indice')
plt.ylabel('$H$ $(A/m)$')
plt.grid()


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
plt.figure(figsize=(6,3.5),constrained_layout=True)
plt.axhline(0,0,1,color='k',lw=0.5)
plt.axvline(0,0,1,color='k',lw=0.5)
plt.plot(x_LF_1/1000,y_LF_1,'.-', label='1er rama')
plt.plot(x_LF_1/1000,g_1,label='LF1')
plt.plot(x_LF_2/1000,y_LF_2,'.-',label='2da rama')
plt.plot(x_LF_2/1000,g_2,label='LF2')
plt.text(31,-1.5e-6,'$\mu = \chi_0 \cdot H+n$',bbox=dict(alpha=0.6))
plt.legend()
plt.grid()
plt.title('Ajuste lineal a bajo campo',fontsize=14)

plt.ylabel('$\mu$ $(Am^2)$')
plt.xlabel('$H$ $(kA/m)$')
plt.savefig('TF_Ajuste_LF.png',dpi=300)
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


#%% Ciclo con inset
fig,ax =plt.subplots()
ax0 = plt.subplot(111)
ax0.plot(campo,mom_mag,'.-',lw=0.5)
#ax.plot(campo[:len(campo)//2],mom_mag[:len(mom_mag)//2],'.-',lw=0.5)
#ax.plot(campo[len(campo)//2:],mom_mag[len(mom_mag)//2:],'.-',lw=0.5)
#ax0.axhline(0,0,1,color='k',lw=0.2)
#x0.axvline(0,0,1,color='k',lw=0.2)
#ax.axhline(max(mom_mag),0,1,color='k',lw=0.7)
#ax.axvline(campo[82],0,1,color='r',lw=0.7)

#plt.xlim(-0.4e6,0.4e6)
#plt.ylim(-2.5e-6,2.5e-6)

axin1 = ax0.inset_axes([1.1,0.3, 0.5,0.5])
axin1.plot(campo[44:64],mom_mag[44:64] ,'-D',c='#2ca02c')
axin1.grid()
ax0.indicate_inset_zoom(axin1)
#axin1.yaxis.tick_right()

#plt.setp(axin1.get_xticklabels(),visible=False)
plt.xlabel('$H$ $(A/m)$')
plt.ylabel('$\mu$ $(Am^2)$')
plt.title('Coloide de NPM de Fe en hexano')
plt.grid()

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
plt.legend(ncol=2)
plt.text(2e7,5.3e-6,'$\mu = (1- a/H)\cdot \mu_s + C\cdot H$',bbox=dict(alpha=0.6))
plt.grid()
plt.ylabel('$\mu$ $(Am^2)$')
plt.xlabel('$H$ $(A/m)$')
plt.title('Ajuste no lineal para altos campos',fontsize=14)
plt.savefig('TF_Ajuste_HF12.png',dpi=300)
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
plt.plot(x_HF_3,y_HF_3,'.-', label='1er rama')
plt.plot(x_HF_3,h_3,label='HF3')
plt.plot(x_HF_4,y_HF_4,'.-',label='2da rama')
plt.plot(x_HF_4,h_4,label='HF4')
plt.legend(ncol=2)
plt.text(-3.3e7,-5.3e-6,'$\mu = (1- a/H)\cdot \mu_s + C\cdot H$',bbox=dict(alpha=0.6))
plt.grid()
plt.ylabel('$\mu$ $(Am^2)$')
plt.xlabel('$H$ $(A/m)$')
#plt.title('Ajuste no lineal para altos campos',fontsize=14)
plt.savefig('TF_Ajuste_HF34.png',dpi=300)
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

#%%
a=np.mean([a1,a2, np.abs(a3) ,np.abs(a4)])
mu_s=np.mean([mu_s1,mu_s2,np.abs(mu_s3),np.abs(mu_s4)])
C=np.mean([C1,C2,C3,C4])
chi_0 = np.mean([m1,m2])

kB=1.38e-24
T=np.mean(temperatura)
mu0 = pi*4e-7
mu_B = 9.27e-24 #A/m

mu_p = kB*T/(mu0*a)
N = mu_s/mu_p
mu_p2 = 3*kB*T*(chi_0-C)/(N*mu0)

#%%
print(''' 
a     {:.4e} m/A
mu_s  {:.4e} Am^2
C     {:.4e} m^3
chi_0 {:.4e} m^3
mu_p  {:.4e} Am^2 = {:.2f} mu_B
N     {:.4e} particulas
mu_p2 {:.4e} (Am^2)^2
'''.format(a,mu_s,C,chi_0,mu_p,mu_p/mu_B,N,mu_p2))
# %%
var = mu_p2 - mu_p**2
SD = np.sqrt(mu_p2-mu_p**2)


print('''Varianza : {:.2e} A^2m^4 = {:.2f} mu_B^2 
SD = {:.2e} Am^2 = {:.2f} mu_B'''.format(var, var/mu_B**2,SD,SD/mu_B))
#%%

fig,ax = plt.subplots(figsize=(6,7),constrained_layout=True)
ax = plt.subplot(211)
ax.plot(x_HF_1,y_HF_1,'.-', label='1er rama')
ax.plot(x_HF_1,h_1,label='HF1')
ax.plot(x_HF_2,y_HF_2,'.-',label='2da rama')
ax.plot(x_HF_2,h_2,label='HF2')
ax.legend(ncol=2)
ax.text(2.3e7,5.2e-6,'$\mu = (1- a/H)\cdot \mu_s + C\cdot H$',bbox=dict(alpha=0.6))
ax.grid()
ax.set_ylabel('$\mu$ $(Am^2)$')

ax2= plt.subplot(212)
ax2.plot(x_HF_3,y_HF_3,'.-', label='1er rama')
ax2.plot(x_HF_3,h_3,label='HF3')
ax2.plot(x_HF_4,y_HF_4,'.-',label='2da rama')
ax2.plot(x_HF_4,h_4,label='HF4')
ax2.legend(ncol=2)
ax2.text(-3.3e7,-5.3e-6,'$\mu = (1- a/H)\cdot \mu_s + C\cdot H$',bbox=dict(alpha=0.6))
ax2.grid()

plt.ylabel('$\mu$ $(Am^2)$')
plt.xlabel('$H$ $(A/m)$')
plt.suptitle('Ajuste no lineal para altos campos',fontsize=14)
plt.savefig('TF_Ajuste_HF.png',dpi=300)
plt.show()


