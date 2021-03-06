#%%
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

#%%
data = np.loadtxt('datos_tema_6.dat',skiprows=1)
tiempo = data[:,0] #[s]
temperatura = data[:,1]#[K]
campo = data[:,2]*1000/(4*pi)#[A/m]
mom_mag =  data[:,3]/1000#[Am^2]
#%%
# Ciclo
fig,ax=plt.subplots()
ax.plot(campo,mom_mag,'.-',lw=0.6,label='T = 220 K')
#ax.plot(campo[:len(campo)//2],mom_mag[:len(campo)//2],'.-',lw=0.7,label='1')
#ax.plot(campo[len(campo)//2:],mom_mag[len(campo)//2:],'.-',lw=0.7,label='2')
#ax.plot(campo[:len(campo)//2]/1000,mom_mag[:len(mom_mag)//2],marker=r'$\swarrow$',ms=10,lw=0.5,label='1')
#ax.plot(campo[len(campo)//2:]/1000,mom_mag[len(mom_mag)//2:],marker=r'$\nearrow$',ms=10,lw=0.5,label='2')
#ax.axvline(0,0,1,color='k',lw=0.5)
#ax.axhline(0,0,1,color='k',lw=0.5)
#ax.axhline(max(mom_mag),0,1,color='k',lw=0.7)
#ax.axvline(campo[82],0,1,color='r',lw=0.7)
ax.legend()
plt.xlabel('$H$ $(A/m)$')
plt.ylabel('$\mu$ $(Am^2)$')
plt.title('Coloide de NPM de Fe en hexano')
plt.grid()
plt.xlim(0,4e6)
plt.ylim(0,8.2e-6)
plt.tight_layout()
plt.savefig('TF_ciclo_crudo2.png',dpi=400)
#plt.show()
#%% 
# D1 = 1308.01
# D2 = 1253.32

#Compenso la coercitividad de cada rama
campo[:108]=campo[:108]-1220.3063164488021#1217.70595
campo[109:]=campo[109:]+1280.2588624865223#1285.58755
#%%
fig,ax=plt.subplots()
#ax.plot(campo,mom_mag,'.-',lw=0.7)
ax.plot(campo[:len(campo)//2],mom_mag[:len(mom_mag)//2],'.-',lw=0.5,label='1')
ax.plot(campo[len(campo)//2:],mom_mag[len(mom_mag)//2:],'.-',lw=0.5,label='2')
ax.axhline(0,0,1,color='k',lw=0.2)
ax.axvline(0,0,1,color='k',lw=0.2)
#ax.axhline(max(mom_mag),0,1,color='k',lw=0.7)
#ax.axvline(campo[82],0,1,color='r',lw=0.7)
ax.legend()
plt.xlabel('$H$ $(A/m)$')
plt.ylabel('$\mu$ $(Am^2)$')
plt.title('Coloide de NPM de Fe en hexano\nRemanencia compensada')
plt.grid()
plt.xlim(min(campo),0e4)
plt.ylim(-8e-6,0)
plt.show()

#%% seleccion del campo para cada ajuste
plt.plot(campo,'.-')
plt.plot(range(0,30),campo[0:30],'o')
plt.plot(range(0,30),campo[0:30],'o')
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
#%% 
# Ajuste Bajo campo (LF) y a alto campo (HF)
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def ajuste_LF(x,a,b):
    return a*x+b
#%%
x_LF_1 = campo[43:64]#1er rama
y_LF_1 = mom_mag[43:64]

x_LF_2 = campo[152:173]#2da rama
y_LF_2 = mom_mag[152:173]
 
param_1,p1_cov = curve_fit(ajuste_LF,x_LF_1,y_LF_1)
param_2,p2_cov = curve_fit(ajuste_LF,x_LF_2,y_LF_2)

m1,n1=param_1
m1_err,n1_err = np.sqrt(np.diag(p1_cov))
print('mu = {:.2e}*H + {:.2e}'.format(m1, n1))
m2,n2=param_2
m2_err,n2_err = np.sqrt(np.diag(p2_cov))
print('mu = {:.2e}*H + {:.2e}'.format(m2, n2))
g_1 = ajuste_LF(x_LF_1,m1,n1)
g_2 = ajuste_LF(x_LF_2,m2,n2)

#%% 
# Figura Ajuste LF 
plt.figure(figsize=(6,3.5),constrained_layout=True)
plt.axhline(0,0,1,color='k',lw=0.5)
plt.axvline(0,0,1,color='k',lw=0.5)
plt.plot(x_LF_1/1000,y_LF_1,'o', label='1er rama')
plt.plot(x_LF_1/1000,g_1,label='LF1 (R$^2$ = {:.3f})'.format(r2_score(y_LF_1,g_1)),lw=0.9)
plt.plot(x_LF_2/1000,y_LF_2,'D',label='2da rama')
plt.plot(x_LF_2/1000,g_2,'-.',label='LF2 (R$^2$ = {:.3f})'.format(r2_score(y_LF_2,g_2)),lw=0.9)
#plt.axvline(-1.302,0.5,1)
plt.text(3.1,-1.5e-6,'$\mu = \chi_0 \cdot H+n$',bbox=dict(alpha=0.6))
plt.legend()
plt.grid()
plt.title('Ajuste lineal a bajo campo',fontsize=14)

plt.ylabel('$\mu$ $(Am^2)$')
plt.xlabel('$H$ $(kA/m)$')
#plt.savefig('TF_Ajuste_LF.jpeg',dpi=300)
plt.show()

print('''Pendiente: 
        m1={:.2e} +/- {:.2e} m^3
        m2={:.2e} +/- {:.2e} m^3
Ordenada:
        n1={:.2e} +/- {:.2e} Am^2
        n2={:.2e} +/- {:.2e} Am^2
Calidad del ajuste:
        R1^2= {:.4f} 
        R2^2= {:.4f} 

'''.format(m1,m1_err,m2,m2_err,n1,n1_err,
n2,n2_err,r2_score(y_LF_1,g_1),r2_score(y_LF_2,g_2)))


#%% Ciclo con inset
fig =  plt.figure(figsize=(7,3.5),constrained_layout=True)
ax0 = fig.add_subplot(1,1,1)
#ax0.plot(campo,mom_mag,'.-',lw=0.5)
ax0.plot(campo[:len(campo)//2],mom_mag[:len(mom_mag)//2],'.-',lw=0.5,label='1')
ax0.plot(campo[len(campo)//2:],mom_mag[len(mom_mag)//2:],'.-',lw=0.5,label='2')
ax0.legend()
#plt.xlim(-0.4e6,0.4e6)
#plt.ylim(-2.5e-6,2.5e-6)

axin1 = ax0.inset_axes([1.1,0.1, 0.5,0.85])
axin1.plot(campo[43:64],mom_mag[43:64] ,'.-',label='1')
axin1.plot(campo[152:173],mom_mag[152:173] ,'.-',label='2')
axin1.grid()
ax0.indicate_inset_zoom(axin1,alpha=1)
axin1.yaxis.tick_right()
axin1.legend()
axin1.set_xlabel('$H$ $(A/m)$')
axin1.set_title('Zona de bajo campo',fontsize=10)

#plt.setp(axin1.get_xticklabels(),visible=False)
plt.xlabel('$H$ $(A/m)$')
plt.ylabel('$\mu$ $(Am^2)$')
plt.suptitle('Coloide de NPM de Fe en hexano',ha='center',fontsize=15)
plt.grid()
#plt.autoscale
#plt.savefig('Ciclo_inset_sin_coerc.jpg',dpi=300)
###################################
#%% 
# Altos campos - Ajuste No lineal
def ajuste_HF(H,a,mu_s,C):
    return (1-a/H)*mu_s + C*H

#1er rama
x_HF_1 = np.flip(campo[0:25])
y_HF_1 = np.flip(mom_mag[0:25])
#2da rama
x_HF_2 = campo[193:218]
y_HF_2 = mom_mag[193:218]
 
parametros_1,cov_1 = curve_fit(ajuste_HF,x_HF_1,y_HF_1)
parametros_2,cov_2 = curve_fit(ajuste_HF,x_HF_2,y_HF_2)

a1,mu_s1,C1=parametros_1
a2,mu_s2,C2=parametros_2

a1_err,mu_s1_err,C1_err=np.sqrt(np.diag(cov_1))
a2_err,mu_s2_err,C2_err=np.sqrt(np.diag(cov_2))

h_1=ajuste_HF(x_HF_1,a1,mu_s1,C1)
h_2=ajuste_HF(x_HF_2,a2,mu_s2,C2)

plt.figure()
plt.plot(x_HF_1,y_HF_1,'o', label='1er rama')
plt.plot(x_HF_1,h_1,label='HF1(R$^2$ = {:.3f})'.format(r2_score(y_HF_1,h_1)),lw=0.8)
plt.plot(x_HF_2,y_HF_2,'D',label='2da rama')
plt.plot(x_HF_2,h_2,'-.',label='HF2 (R$^2$ = {:.3f})'.format(r2_score(y_HF_2,h_2)),lw=0.8)
plt.legend(ncol=2)
plt.text(2.35e6,6.25e-6,r'$\mu=\mu_s\cdot\left(1- \frac{a}{H}\right) + C\cdot H$',bbox=dict(alpha=0.6),ha='center',va='center')
plt.grid()
plt.ylabel('$\mu$ $(Am^2)$')
plt.xlabel('$H$ $(A/m)$')
plt.title('Ajuste no lineal para altos campos',fontsize=14)
plt.savefig('TF_Ajuste_HF12.jpeg',dpi=300)
plt.show()

print(''' 
a1= {:.2e} +/- {:.2e} m/A
a2= {:.2e} +/- {:.2e} m/A

mu_s1= {:.2e} +/- {:.2e} Am^2
mu_s2= {:.2e} +/- {:.2e} Am^2

C1= {:.2e} +/- {:.2e} m^3
C2= {:.2e} +/- {:.2e} m^3

R1^2= {:.4f} 
R2^2= {:.4f}'''.format(a1,a1_err,a2,a2_err,mu_s1,mu_s1_err,
mu_s2,mu_s2_err,C1,C1_err,C2,C2_err,
r2_score(y_HF_1,h_1),r2_score(y_HF_2,h_2)))

#%% 
# Campos negativos
#1er rama
x_HF_3 = -campo[84:109]
y_HF_3 = -mom_mag[84:109]

#2da rama
x_HF_4 = -np.flip(campo[109:134])
y_HF_4 = -np.flip(mom_mag[109:134])
 
parametros_3,cov_3 = curve_fit(ajuste_HF,x_HF_3,y_HF_3)
parametros_4,cov_4 = curve_fit(ajuste_HF,x_HF_4,y_HF_4)

a3,mu_s3,C3=parametros_3
a4,mu_s4,C4=parametros_4

a3_err,mu_s3_err,C3_err=np.sqrt(np.diag(cov_3))
a4_err,mu_s4_err,C4_err=np.sqrt(np.diag(cov_4))

h_3 = ajuste_HF(x_HF_3,a3,mu_s3,C3)
h_4 = ajuste_HF(x_HF_4,a4,mu_s4,C4)
plt.figure()
plt.plot(-x_HF_3,-y_HF_3,'o', label='1er rama')
plt.plot(-x_HF_3,-h_3,label='HF3 (R$^2$ = {:.3f})'.format(r2_score(y_HF_3,h_3)),lw=0.8)
plt.plot(-x_HF_4,-y_HF_4,'D',label='2da rama')
plt.plot(-x_HF_4,-h_4,'-.',label='HF4 (R$^2$ = {:.3f})'.format(r2_score(y_HF_1,h_1)),lw=0.8)
plt.legend(ncol=2)
plt.text(-2.35e6,-6.25e-6,r'$\mu=\mu_s\cdot\left(1- \frac{a}{H}\right) + C\cdot H$',bbox=dict(alpha=0.6),ha='center',va='center')
plt.grid()
plt.ylabel('$\mu$ $(Am^2)$')
plt.xlabel('$H$ $(A/m)$')
#plt.title('Ajuste no lineal para altos campos',fontsize=14)
plt.savefig('TF_Ajuste_HF34.jpeg',dpi=300)
plt.show()
print('''
a3= {:.2e} +/- {:.2e} m/A
a4= {:.2e} +/- {:.2e} m/A

mu_s3= {:.2e} +/- {:.2e} Am^4
mu_s4= {:.2e} +/- {:.2e} Am^4

C3= {:.2e} +/- {:.2e} m^3
C4= {:.2e} +/- {:.2e} m^3

R3^4= {:.4f} 
R4^4= {:.4f}'''.format(a3,a3_err,a4,a4_err,mu_s3,mu_s3_err,
mu_s4,mu_s4_err,C3,C3_err,C4,C4_err,
r2_score(y_HF_3,h_3),r2_score(y_HF_4,h_4)))

#%% 
# Estadistica final

from uncertainties import unumpy

a = np.mean(unumpy.uarray([a1, a2,a3 ,a4],[a1_err, a2_err,a3_err,a4_err]))
mu_s=np.mean(unumpy.uarray([mu_s1,mu_s2,np.abs(mu_s3),np.abs(mu_s4)],[mu_s1_err,mu_s2_err,mu_s3_err,mu_s4_err]))
C=np.mean(unumpy.uarray([C1,C2,C3,C4],[C1_err,C2_err,C3_err,C4_err]))

chi_0 = np.mean(unumpy.uarray([m1,m2],[m1_err,m2_err]))

kB=1.38e-23 #J/K
T=unumpy.uarray(np.mean(temperatura),np.std(temperatura))

mu0 = pi*4e-7
mu_B = 9.27e-24 #A/m

mu_p = (kB*T)/(mu0*a)
N = mu_s/mu_p
mu_p2 = (3*kB*T)*(chi_0-C)/(N*mu0)

#Densidad
rho = 5.18 #g/cm^3 
#Magnetizacion masica
sigma = 88.1 #emu/g
#Magnetizacion volumetrica
M = rho*sigma #emu/cm^3
M *= 1e3 #A/m
D = ((6/pi)*(mu_p/M))**(1/3) #m
D *=1e9 #nm
print(''' 
a:     {:.2e} m/A
mu_s:  {:.3e} Am^2 = {:.4} mu_B
C:     {:.2e} m^3
chi_0: {:.2e} m^3
mu_p:  {:.3e} Am^2 = {:.3f} mu_B
N:     {:.2e} particulas
mu_p2: {:.2e} (Am^2)^2 = {:.2e} mu_B^2
D:     {:.2f} nm 
'''.format(a,mu_s,mu_s/mu_B,C,chi_0,mu_p,mu_p/mu_B,N,mu_p2,mu_p2/(mu_B**2),D))

var = unumpy.nominal_values(mu_p2) - unumpy.nominal_values(mu_p)**2
SD =unumpy.sqrt(var)

print('''Varianza : {:.2e} A^2m^4 = {:.2e} mu_B^2 
SD = {:.2e} Am^2 = {:.2e} mu_B'''.format(var, var/mu_B**2,SD,SD/mu_B))
#%% 
# Plot doble para pdf

fig,ax = plt.subplots(figsize=(6,7),constrained_layout=True)
ax = plt.subplot(211)
ax.plot(x_HF_1,y_HF_1,'o', label='1?? rama')
ax.plot(x_HF_1,h_1,label='HF1 - R$^2$ = {:.3f}'.format(r2_score(y_HF_1,h_1)),lw=1)
ax.plot(x_HF_2,y_HF_2,'D',label='2?? rama')
ax.plot(x_HF_2,h_2,'-.',label='HF2 - R$^2$ = {:.3f}'.format(r2_score(y_HF_1,h_1)),lw=1)
ax.legend(ncol=2)
ax.text(2.35e6,6.25e-6,r'$\mu=\mu_s\cdot\left(1- \frac{a}{H}\right) + C\cdot H$',bbox=dict(alpha=0.6),ha='center',va='center')
ax.grid()
ax.set_ylabel('$\mu$ $(Am^2)$')

ax2= plt.subplot(212)
ax2.plot(-x_HF_3,-y_HF_3,'o', label='1?? rama')
ax2.plot(-x_HF_3,-h_3,label='HF3 - R$^2$ = {:.3f}'.format(r2_score(y_HF_3,h_3)))
ax2.plot(-x_HF_4,-y_HF_4,'D',label='2?? rama')
ax2.plot(-x_HF_4,-h_4,'-.',label='HF4 - R$^2$ = {:.3f}'.format(r2_score(y_HF_4,h_4)))
ax2.legend(ncol=2)
ax2.text(-2.35e6,-6.25e-6,r'$\mu=\mu_s\cdot\left(1- \frac{a}{H}\right) + C\cdot H$',bbox=dict(alpha=0.6),ha='center',va='center')
ax2.grid()

plt.ylabel('$\mu$ $(Am^2)$')
plt.xlabel('$H$ $(A/m)$')
plt.suptitle('  Ajuste no lineal para altos campos',fontsize=15)
plt.savefig('TF_Ajuste_HF.png',dpi=300)
plt.show()


# %%
#%% 
# 
# Seleccion del campo para cada ajuste
plt.plot(campo,'-',lw=0.5)
plt.plot(range(0,28),campo[0:28],'.',label='NL1')
plt.plot(range(79,109),campo[79:109],'.',label='NL1')

plt.plot(range(43,64),campo[43:64],'.',label='L1')

plt.plot(range(193,218),campo[193:218],'.',label="NL2")
plt.plot(range(109,138),campo[109:138],'.',label='NL2')

plt.plot(range(152,173),campo[152:173],'.',label='L2')

plt.legend(ncol=2)
plt.title('Campo Magnetico')
plt.xlabel('Indice')
plt.ylabel('$H$ $(A/m)$')
plt.grid()
plt.show()

#%%
plt.plot(tiempo,campo,'-',lw=0.5)
plt.plot(tiempo[0:28],campo[0:28],'o',label='NL1')
plt.plot(tiempo[79:109],campo[79:109],'o',label='NL1')
plt.plot(tiempo[43:64],campo[43:64],'H',label='L1')
plt.plot(tiempo[193:218],campo[193:218],'D',label="NL2")
plt.plot(tiempo[109:138],campo[109:138],'D',label='NL2')
plt.plot(tiempo[152:173],campo[152:173],'H',label='L2')
plt.legend(ncol=2)
plt.title('Campo Magnetico')
plt.xlabel('tiempo (s)')
plt.ylabel('$H$ $(A/m)$')
plt.grid()

#plt.savefig('H_vs_t.jpeg',dpi=300)
plt.show()
#%%
plt.plot(campo[:len(campo)//2],mom_mag[:len(mom_mag)//2],'-',lw=0.5)
plt.plot(campo[0:28],mom_mag[0:28],'o',label='NL1')
plt.plot(campo[79:109],mom_mag[79:109],'o',label='NL1')
plt.plot(campo[43:64],mom_mag[43:64],'H',label='L1')
#plt.plot(campo[193:218],mom_mag[193:218],'D',c='tab:purple',label="NL2")
#plt.plot(campo[109:138],mom_mag[109:138],'D',c='tab:brown',label='NL2')
#plt.plot(campo[152:173],mom_mag[152:173],'H',c='tab:pink',label='L2')
plt.legend(ncol=1)
plt.xlabel('$H$ $(A/m)$')
plt.ylabel('$\mu$ $(Am^2)$')
plt.grid()

#plt.savefig('Ciclo_rama1.jpeg',dpi=300)
plt.show()
#%%
#1 rama
x_HF_1 = np.flip(campo[0:28])
y_HF_1 = np.flip(mom_mag[0:28])
#2da rama
x_HF_2 = campo[188:218]
y_HF_2 = mom_mag[188:218]

#1er rama
x_HF_3 = -campo[79:109]
y_HF_3 = -mom_mag[79:109]
#2da rama
x_HF_4 = -np.flip(campo[109:138])
y_HF_4 = -np.flip(mom_mag[109:138])

#1er rama
x_LF_1 = campo[43:64]
y_LF_1 = mom_mag[43:64]
#2da rama
x_LF_2 = campo[152:173]
y_LF_2 = mom_mag[152:173]
#%%
# Ciclo Presentacion
from matplotlib.lines import Line2D
marker_style = dict(color='tab:blue', linestyle='-', marker='.',ms=8)

marker_style2 = dict(color='tab:green', linestyle='-', marker='.',ms=7)

fig,ax=plt.subplots()
#ax.plot(campo[:len(campo)//2],mom_mag[:len(campo)//2],'o-',lw=0.7,label='1')
#ax.plot(campo[len(campo)//2:],mom_mag[len(campo)//2:],'D-',lw=0.7,label='2')
ax.plot(campo[:len(campo)//2],mom_mag[:len(campo)//2],fillstyle='full', **marker_style,lw=0.5,label='1er rama')

ax.plot(campo[len(campo)//2:],mom_mag[len(campo)//2:],fillstyle='right', **marker_style2,lw=0.5,label='2da rama')
#ax.plot(campo,mom_mag,'.-',lw=0.5,label='2')
#ax.axvline(0,0,1,color='k',lw=0.2)
#ax.axhline(max(mom_mag),0,1,color='k',lw=0.7)
#ax.axvline(campo[82],0,1,color='r',lw=0.7)
ax.legend()
plt.xlabel('$H$ $(A/m)$')
plt.ylabel('$\mu$ $(Am^2)$')
plt.title('Coloide de NPM de Fe en hexano')
plt.grid()




#plt.xlim(-1e1,1e1)
#plt.ylim(-2.0e-6,2.0e-6)
plt.savefig('TF_ciclo.jpeg',dpi=300)
#plt.show()
# %%
