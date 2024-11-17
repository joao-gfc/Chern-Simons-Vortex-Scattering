'''
This code simulates the scattering between a Chern-Simons vortex and a half-BPS preserving magnetic impurity
'''
# We start declaring the relevant libraries
from mpi4py import MPI #This is for paralelization on the cluster where we ran the simulations
import numpy as np
from scipy.integrate import solve_bvp
from scipy.integrate import solve_ivp
from scipy import interpolate, optimize
import sys

#These are paralelization data. Each CPU node gets a different rank number and, therefore, can run with different parameters.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#Some parameters
L=16.0                        #box size in each axis
N=512                         #number of gridpoints in each axis
a=0.0625                      #grid spacing is 2L/N
kappa=2                       #this is the kappa parameter in chern-simons theory
x = np.linspace(-L, L, N+1)   #this is an array for the x-axis gridpoints
xv, yv = np.meshgrid(x, x)    #this is a mesh (matrix) with the corresponding x and y values
r2=xv**2+yv**2                #this is another mesh (matrix) with the corresponding values of r^2=x^2+y^2

coef1=float(sys.argv[1])      #this is the first impurity parameter that I get as an argument from sys.argv
coef2=1.0                     #this is the second impurity paramters. This time it is hard-coded.

#The impurity and its derivatives are defined below. It is a Gaussian profile.
def sigma(r,coef1,coef2):
  #gaussian impurity
  return coef1*np.exp(-coef2*r**2)

def sigma_x(x,r,coef1,coef2):
  #partial derivative with respect to x of sigma
  return -2*x*coef1*coef2*np.exp(-coef2*r**2)

def sigma_y(y,r,coef1,coef2):
  #partial derivative with respect to y of sigma
  return -2*y*coef1*coef2*np.exp(-coef2*r**2)

#The following loops are used to contruct matrices representing the values of sigma and its derivatives at the gridpoints. 
sigmar=[]
sigmar_x=[]
sigmar_y=[]
for i in range(N+1):
  x=-L+i*a
  tmp1=[]
  tmp2=[]
  tmp3=[]
  for j in range(N+1):
    y=-L+j*a
    r=np.sqrt(x**2+y**2)
    tmp1.append(sigma(r,coef1,coef2))
    tmp2.append(sigma_x(x,r,coef1,coef2))
    tmp3.append(sigma_y(y,r,coef1,coef2))
  sigmar.append(tmp1)
  sigmar_x.append(tmp2)
  sigmar_y.append(tmp3)

sigmar=np.array(sigmar)
sigmar_x=np.array(sigmar_x)
sigmar_y=np.array(sigmar_y)

#Here I define the Chern-Simons Potential. Basically it is a phi^6 potential.
def V(phi):
  return 0.25*phi**2*(1+sigmar-phi**2)**2

'''
     
  Our goal now is to compute the vortex solution, that is, the solution with winding number one.

'''

def fun2(x, y):
  #this function represents the Taubes equation for BPS vortices
  return np.vstack((y[1], np.exp(y[0])*(-1.0+np.exp(y[0]))-y[1]/x))
  
def bc2(ya, yb):
  #this is the boundary condition for a vortex
  return np.array([ya[1]-200, yb[0]])

# This is a grid for taubes equation. We are using spherical coordinates. So r>0. 
# To avoid the Taubes eq. singularity at r=0, we start at r=epsilon=0.01. This leads to an error of order epsilon^2. 
r_axis = np.linspace(0.01, 100.0, 10000)

#This is an initial guess for the vortex solution.
y0=np.stack((2*np.log(np.tanh(0.4*r_axis)),0.8/(np.sinh(0.4*r_axis)*np.cosh(0.4*r_axis))))

res2 = solve_bvp(fun2, bc2, r_axis, y0, tol=1e-9) #solution is computed here

#Now we prepare to boost the solution. And construct a matrix with the field values at the gridpoints. 
v=float(sys.argv[2])        #initial vortex velocity
gamma=1.0/np.sqrt(1-v**2)   #corresponding gamma values
E0=gamma*2*np.pi            #Thus, the initial vortex energy is 2*pi*gamma
Es=[E0]                     #A list to store the energy values

#The initial charge is 1 and its stored in a list
Qs=[1.0]                    

#posição inicial
x0=-8.0              #initial x position
y0=-12.0+0.5*rank    #intiial y position
pos=[[x0,y0]]        #list to store the positions

#Campos iniciais
phiL=[] #scalar field
piL=[]  #conjugate momentum to phi
A1L=[]  #gauge field 1
A2L=[]  #gauge field 2
A0L=[]  #gauge field 0

#This part is a little tricky, but as I said, I am just contructing a matrix with the fields values at the gridpoints. 
#The formulas a lengthy because you need to carefully implement the boost equations.
for i in range(N+1):
  x=-L+i*a                                # x value
  tmp1=[]
  tmp2=[]
  tmp3=[]
  tmp4=[]
  tmp5=[]
  for j in range(N+1):
    y=-L+j*a                              # y value
    r2=x**2+y**2			  # r^2=x^2+y^2
    r=np.sqrt(r2)                         # value of the actual radius of the grid.
    s2=gamma**2*(x-x0)**2+(y-y0)**2       # "effective" radius for a boost vortex s. 
    s=np.sqrt(s2)                         # s is lorentz contracted in the x axis.
    #Sometimes you cannot just put s=0 or r=0. Insted, you have to carefully take the limits in the boost formulas as folows.
    if(s2<1e-6):
      tmp1.append(0.0)
      tmp2.append(-v*gamma*np.exp(0.5*res2.sol(0.01)[0])*0.5*res2.sol(0.01)[1])
      tmp3.append(0.0)
      tmp4.append(0.0)
      tmp5.append(0.5)
    #Now we implement the boost formulas for s and r nonzero.
    else:
      tmp1.append(np.exp(0.5*res2.sol(s)[0])*(gamma*(x-x0)/s+1j*(y-y0)/s))
      tmp2.append( 0.5j*gamma*(1-np.exp(res2.sol(s)[0])) * np.exp(0.5*res2.sol(s)[0])*(gamma*(x-x0)/s+1j*(y-y0)/s)
                 - v*(np.exp(0.5*res2.sol(s)[0])*0.5*res2.sol(s)[1]*gamma**2*(x-x0)/s*(gamma*(x-x0)/s+1j*(y-y0)/s)
                      +np.exp(0.5*res2.sol(s)[0])*(gamma/s-gamma**2*(x-x0)*(gamma*(x-x0)+1j*(y-y0))/s**3))         
                 + 1j*v*gamma*(y-y0)*(0.5*res2.sol(s)[1]/s-1/s2) * np.exp(0.5*res2.sol(s)[0])*(gamma*(x-x0)/s+1j*(y-y0)/s))
      tmp3.append(      -(y-y0)*(0.5*res2.sol(s)[1]/s-1/s2))
      tmp4.append( gamma*(x-x0)*(0.5*res2.sol(s)[1]/s-1/s2))
      tmp5.append(0.5*(1-np.exp(res2.sol(s)[0])))
  phiL.append(tmp1)
  piL.append(tmp2)
  A1L.append(tmp3)
  A2L.append(tmp4)
  A0L.append(tmp5)

phiL=np.array(phiL)
piL=np.array(piL)
A1L=np.array(A1L)
A2L=np.array(A2L)
A0L=np.array(A0L)

#There is one last step in the gauge field boost, as shown below.
A1L_new=gamma*A1L-v*gamma*np.array(A0L)
A0L_new=gamma*A0L-v*gamma*np.array(A1L)
A1L=A1L_new
A0L=A0L_new

'''
     
  Our goal now is to compute the impurity solution, that is, the solution with winding number zero.

'''

def fun1(x, y):
  #Here we write taubes equation
  return np.vstack((y[1], np.exp(y[0])*(-1.0-sigma(x,coef1,coef2)+np.exp(y[0]))-y[1]/x))
  
def bc1(ya, yb):
    #Boundary condition for Taubes equations
    return np.array([ya[1], yb[0]])

# This is a grid for taubes equation. We are using spherical coordinates. So r>0. 
# We add a very small point r=1e-6. This seems to decrease the error in truncating the r axis.
r_axis=np.concatenate(([1e-6],r_axis))

#The initial for the impurity. This choice may not be appropriate for a different impurity.
initial_guess2 = np.vstack((0.4/np.cosh(r_axis),-0.4*np.tanh(r_axis)/np.cosh(r_axis)))

#Solving for the impurity
res1 = solve_bvp(fun1, bc1, r_axis, initial_guess2, tol=1e-9)

#Now we construct the field matrices for the impurity solutions, once more. 
#We just have to translate the taubes field back to phi and A.
phiI=[]
piI=[]
A1I=[]
A2I=[]
A0I=[]

for i in range(N+1):
  x=-L+i*a
  tmp1=[]
  tmp2=[]
  tmp3=[]
  tmp4=[]
  tmp5=[]
  for j in range(N+1):
    y=-L+j*a
    r2=x**2+y**2
    r=np.sqrt(r2)
    #The first part is the limit as r->0.
    if(r2<1e-6):
      tmp1.append(np.exp(0.5*res1.sol(0.0)[0]))
      tmp2.append( 0.5j*(1+coef1-np.exp(res1.sol(0.0)[0])) * np.exp(0.5*res1.sol(0.0)[0]) )
      tmp3.append(0.0)
      tmp4.append(0.0)
      tmp5.append(0.5*(1+coef1-np.exp(res1.sol(0.0)[0])))
    #These equations can be found in relatively easily found in the literature (I think the original work for CS is by Jackiw).
    else:
      tmp1.append(np.exp(0.5*res1.sol(r)[0]))
      tmp2.append( 0.5j*(1+sigma(r,coef1,coef2)-np.exp(res1.sol(r)[0])) * np.exp(0.5*res1.sol(r)[0]) )
      tmp3.append(-y*(0.5*res1.sol(r)[1]/r))
      tmp4.append( x*(0.5*res1.sol(r)[1]/r))
      tmp5.append( 0.5 *(1+sigma(r,coef1,coef2)-np.exp(res1.sol(r)[0])) )
  phiI.append(tmp1)
  piI.append(tmp2)
  A1I.append(tmp3)
  A2I.append(tmp4)
  A0I.append(tmp5)

phiI=np.array(phiI)
piI=np.array(piI)
A1I=np.array(A1I)
A2I=np.array(A2I)
A0I=np.array(A0I)

'''
     
  Now we glue the impurity and vortex solutions via the abrikosov anstaz.

'''

phi=phiL*phiI
pi=piL*phiI+piI*phiL
A1=A1L+A1I
A2=A2L+A2I
A0=A0L+A0I

'''

Okay, Initial condition is computed
Let us evolve the equations of motions

'''

#This functions implements the equations of motion to be integrated subsequently.
def diffeq(t, y):
  # We have a biiiig array y. We start by separating it in 5 matrices. One for each field.
  phi=np.reshape(y[0:(N+1)*(N+1)], (N+1,N+1))+1j*np.reshape(y[(N+1)*(N+1):2*(N+1)*(N+1)], (N+1,N+1))
  pi=np.reshape(y[2*(N+1)*(N+1):3*(N+1)*(N+1)], (N+1,N+1))+1j*np.reshape(y[3*(N+1)*(N+1):4*(N+1)*(N+1)], (N+1,N+1))
  A0=np.reshape(y[4*(N+1)*(N+1):5*(N+1)*(N+1)], (N+1,N+1))
  A1=np.reshape(y[5*(N+1)*(N+1):6*(N+1)*(N+1)], (N+1,N+1))
  A2=np.reshape(y[6*(N+1)*(N+1):7*(N+1)*(N+1)], (N+1,N+1))

  #Computes the charge density
  rho=-2*np.imag(phi*np.conj(pi))
  
  #Computes partial derivatives using 5-point stencil approximation
  dx_phi=(-np.roll(phi,-2,axis=0)+8*np.roll(phi,-1,axis=0)-8*np.roll(phi,1,axis=0)
          +np.roll(phi,2,axis=0))/(12*a)
  dy_phi=(-np.roll(phi,-2,axis=1)+8*np.roll(phi,-1,axis=1)-8*np.roll(phi,1,axis=1)
          +np.roll(phi,2,axis=1))/(12*a)

  dx_A1=(-np.roll(A1,-2,axis=0)+8*np.roll(A1,-1,axis=0)-8*np.roll(A1,1,axis=0)+np.roll(A1,2,axis=0))/(12*a)
  dy_A2=(-np.roll(A2,-2,axis=1)+8*np.roll(A2,-1,axis=1)-8*np.roll(A2,1,axis=1)+np.roll(A2,2,axis=1))/(12*a)
  dx_A0=(-np.roll(A0,-2,axis=0)+8*np.roll(A0,-1,axis=0)-8*np.roll(A0,1,axis=0)+np.roll(A0,2,axis=0))/(12*a)
  dy_A0=(-np.roll(A0,-2,axis=1)+8*np.roll(A0,-1,axis=1)-8*np.roll(A0,1,axis=1)+np.roll(A0,2,axis=1))/(12*a)

  #Computes covariant derivatives
  Dx_phi=dx_phi+1j*A1*phi
  Dy_phi=dy_phi+1j*A2*phi
  
  #This is the trickiest part of the code. Implementing the boundary conditions. 
  #We chose to se the covariant derivatives to zero at the boundary for phi.
  #We chose to se the PARTIAL derivatives to zero at the boundary for A.
  for j in range(N+1):
    #Due to our choice, these are zero.
    Dx_phi[0,j]=0.0
    Dx_phi[N,j]=0.0
    Dy_phi[j,0]=0.0
    Dy_phi[j,N]=0.0
    #It is good practice to reduce the order to three-point stencil at the boundary, because phi[-1,j] does not exist.
    Dx_phi[1,j]=  (-phi[0,j]+phi[2,j])  /(2*a)+1j*A1[1,j]  *phi[1,j] 
    Dx_phi[N-1,j]=( phi[N,j]-phi[N-2,j])/(2*a)+1j*A1[N-1,j]*phi[N-1,j]
    Dy_phi[j,1]=  (-phi[j,0]+phi[j,2])  /(2*a)+1j*A2[j,1]  *phi[j,1]
    Dy_phi[j,N-1]=( phi[j,N]-phi[j,N-2])/(2*a)+1j*A2[j,N-1]*phi[j,N-1]
    
    #Due to our choices, these are zero
    dx_A1[0,j]= 0.0
    dx_A1[N,j]= 0.0
    dy_A2[j,0]= 0.0
    dy_A2[j,N]= 0.0
    #It is good practice to reduce the order to three-point stencil at the boundary, because A[-1,j] does not exist.
    dx_A1[1,j]=  (-A1[0,j]+A1[2,j]  )/(2*a)  
    dx_A1[N-1,j]=( A1[N,j]-A1[N-2,j])/(2*a)
    dy_A2[j,1]=  (-A2[j,0]+A2[j,2]  )/(2*a)  
    dy_A2[j,N-1]=( A2[j,N]-A2[j,N-2])/(2*a)

    #We do not have more freedom to set these derivatives to zero. So we reduce the order to two-point stencil.   
    dx_A0[0,j]= (-A0[0,j] +A0[1,j]  )/a
    dx_A0[N,j]= ( A0[N,j] -A0[N-1,j])/a
    dy_A0[j,0]= (-A0[j,0] +A0[j,1]  )/a
    dy_A0[j,N]= ( A0[j,N] -A0[j,N-1])/a
    #Here we use three-point stencil again.
    dx_A0[1,j]=  (-A0[0,j]+A0[2,j]  )/(2*a)  
    dx_A0[N-1,j]=( A0[N,j]-A0[N-2,j])/(2*a)
    dy_A0[j,1]=  (-A0[j,0]+A0[j,2]  )/(2*a)  
    dy_A0[j,N-1]=( A0[j,N]-A0[j,N-2])/(2*a)
  
  #Computes second covariant derivative
  D2x_phi=(-np.roll(phi,-2,axis=0)+16*np.roll(phi,-1,axis=0)-30*phi+16*np.roll(phi,1,axis=0)
           -np.roll(phi,2,axis=0))/(12*a**2) - A1**2*phi+2j*A1*dx_phi+1j*dx_A1*phi
  D2y_phi=(-np.roll(phi,-2,axis=1)+16*np.roll(phi,-1,axis=1)-30*phi+16*np.roll(phi,1,axis=1)
           -np.roll(phi,2,axis=1))/(12*a**2) - A2**2*phi+2j*A2*dy_phi+1j*dy_A2*phi
  
  #Boundary condtions
  for j in range(N+1):
    #Two point stencil together with the boundary condtion Dphi=0 leads to these realations.
    D2x_phi[0,j]= Dx_phi[1,j]  /a
    D2x_phi[N,j]=-Dx_phi[N-1,j]/a
    D2y_phi[j,0]= Dy_phi[j,1]  /a
    D2y_phi[j,N]=-Dy_phi[j,N-1]/a
    #Three point stencil together with the boundary condtion Dphi=0 leads to these realations.
    D2x_phi[1,j]  = Dx_phi[2,j]  /(2*a)+1j*A1[1,j]  *Dx_phi[1,j]  
    D2x_phi[N-1,j]=-Dx_phi[N-2,j]/(2*a)+1j*A1[N-1,j]*Dx_phi[N-1,j] 
    D2y_phi[j,1]  = Dy_phi[j,2]  /(2*a)+1j*A2[j,1]  *Dy_phi[j,1]   
    D2y_phi[j,N-1]=-Dy_phi[j,N-2]/(2*a)+1j*A2[j,N-1]*Dy_phi[j,N-1]

  #Computes charge current.
  J1=-2*np.imag(np.conj(phi)*Dx_phi)
  J2=-2*np.imag(np.conj(phi)*Dy_phi)
    
  #Now we compute the actual equations of motion
  dpi=D2x_phi+D2y_phi-1j*A0*pi\
         -0.25*(1+sigmar-np.absolute(phi)**2)*(1+sigmar-3*np.absolute(phi)**2)*phi
  dphi=pi-1j*A0*phi
  
  dA1 = dx_A0 - 0.5*J2 - 0.5*sigmar_x
  dA2 = dy_A0 + 0.5*J1 - 0.5*sigmar_y
  dA0 = dx_A1 + dy_A2

  #And we flatten the equations to return a biiiig y array again.
  return np.concatenate((dphi.real.flatten(),dphi.imag.flatten(),dpi.real.flatten(),dpi.imag.flatten(),
                         dA0.flatten(),dA1.flatten(),dA2.flatten()))
                         
#This is the actual integration of the equations of motion
IC=np.concatenate((phi.real.flatten(),phi.imag.flatten(),pi.real.flatten(),pi.imag.flatten(),
                   A0.flatten(),A1.flatten(),A2.flatten()))
t_eval=[4.0*i for i in range(51)]
sol=solve_ivp(diffeq,[0.0,200.0],IC,t_eval=t_eval,method="RK23") #RK23 seems to be the most stable integrator

'''

# Now with the fields data, we compute the energy, charge and position of the vortex
# Position is the scattering output. Energy and charge is just to make sure that the algorithm is stable.

'''

# Define a function to find zeros of the scalar field within the plaquette
def func(z):
  return abs(interpolate.interpn(points,values,(z[0],z[1])))**2  

for i in range(51):
  y=sol.y[:,i]
  
  #we go from y to matrices again
  phi=np.reshape(y[0:(N+1)*(N+1)], (N+1,N+1))+1j*np.reshape(y[(N+1)*(N+1):2*(N+1)*(N+1)], (N+1,N+1))
  pi=np.reshape(y[2*(N+1)*(N+1):3*(N+1)*(N+1)], (N+1,N+1))+1j*np.reshape(y[3*(N+1)*(N+1):4*(N+1)*(N+1)], (N+1,N+1))
  A0=np.reshape(y[4*(N+1)*(N+1):5*(N+1)*(N+1)], (N+1,N+1))
  A1=np.reshape(y[5*(N+1)*(N+1):6*(N+1)*(N+1)], (N+1,N+1))
  A2=np.reshape(y[6*(N+1)*(N+1):7*(N+1)*(N+1)], (N+1,N+1))
    
  #Calcula a posição
  if (i>0):
    for k in range(N):
      for l in range(N):
        arg1=np.angle(phi[k,l+1]/phi[k,l]    )
        arg2=np.angle(phi[k+1,l+1]/phi[k,l+1]  )
        arg3=np.angle(phi[k+1,l]/phi[k+1,l+1])
        arg4=np.angle(phi[k,l]/phi[k+1,l]  )
        #The argument of the complex number phi rotates by 2pi around a zero
        if(np.abs(arg1+arg2+arg3+arg4) > 1.9*np.pi): #this condition finds the plaquette with a zero.
          x1, x2 = -L+a*k, -L+a*(k+1) 
          y1, y2 = -L+a*l, -L+a*(l+1)
          points=((x1,x2),(y1,y2))
          values=((phi[k,l]  ,phi[k+1,l]),
                  (phi[k,l+1],phi[k+1,l+1]))
          
          # Find a zero of the field within the plaquette using the Newton-Raphson method
          x0 = [(x1 + x2)/2.0, (y1 + y2)/2.0]  # initial guess for the zero
          zero = optimize.minimize(func, x0, bounds=points)
          pos.append([zero.x[0], zero.x[1]])

  #Computes charge density
  rho=-2*np.imag(phi*np.conj(pi))
  
  #Computes partial derivatives using 5-point stencil approximation
  dx_phi=(-np.roll(phi,-2,axis=0)+8*np.roll(phi,-1,axis=0)-8*np.roll(phi,1,axis=0)
          +np.roll(phi,2,axis=0))/(12*a)
  dy_phi=(-np.roll(phi,-2,axis=1)+8*np.roll(phi,-1,axis=1)-8*np.roll(phi,1,axis=1)
          +np.roll(phi,2,axis=1))/(12*a)

  #Computes covariant derivatives
  Dx_phi=dx_phi+1j*A1*phi
  Dy_phi=dy_phi+1j*A2*phi
  
  #Once more, implementing the boundary conditions. 
  #We chose to se the covariant derivatives to zero at the boundary for phi.
  for j in range(N+1):
    #Due to our choice, these are zero.
    Dx_phi[0,j]=0.0
    Dx_phi[N,j]=0.0
    Dy_phi[j,0]=0.0
    Dy_phi[j,N]=0.0
    #It is good practice to reduce the order to three-point stencil at the boundary, because phi[-1,j] does not exist.
    Dx_phi[1,j]=  (-phi[0,j]+phi[2,j])  /(2*a)+1j*A1[1,j]  *phi[1,j] 
    Dx_phi[N-1,j]=( phi[N,j]-phi[N-2,j])/(2*a)+1j*A1[N-1,j]*phi[N-1,j]
    Dy_phi[j,1]=  (-phi[j,0]+phi[j,2])  /(2*a)+1j*A2[j,1]  *phi[j,1]
    Dy_phi[j,N-1]=( phi[j,N]-phi[j,N-2])/(2*a)+1j*A2[j,N-1]*phi[j,N-1]

  # This comes from the energy momentum tensor
  e1=np.absolute(pi)**2
  e2=np.absolute(Dx_phi)**2+np.absolute(Dy_phi)**2
  e3=V(np.absolute(phi))
  e4=-0.5*sigmar*rho

  #adds energy and charge to a list
  Es.append(np.sum(e1+e2+e3+e4)*a**2)
  Qs.append(np.sum(rho)*a**2/(4*np.pi))
  
#Saves charge, energy and position to a file
np.savetxt("en"+str(rank)+".dat",Es)
np.savetxt("charge"+str(rank)+".dat",Qs)
np.savetxt("pos"+str(rank)+".dat",pos)
