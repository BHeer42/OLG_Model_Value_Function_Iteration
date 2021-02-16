# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:16:54 2021

@author: heerburk

AK60_value.py 

simple value function iteration for the value function iteration problem

Ch. 9.1.2 in Heer/Maussner, Dynamic General Equilibrium Modeling: Computational
Methods and Applications, Algorithm 9.1.1

you may adjust the interpolation method (linear, cubic) in line 19

A detailed description of the code can be found on the web page

https://assets.uni-augsburg.de/media/filer_public/b0/4d/b04d79b7-2609-40ac-870b-126aada8e3f4/script_dge_python_11jan2021.html


"""


_VI_method = 2       # 1 --- linear interpolation, 2 --- cubic spline 


# Part 1: import libraries
import numpy as np
from scipy import interpolate
import time
import math
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10,6)

# abbreviations
exp = np.e
log = math.log


# Part 2: define functions


# wage function
def wage_rate(k,l):
    return (1-alp) * k**alp * l**(-alp)

# interest rate function
def interest_rate(k,l):
    return alp * k**(alp - 1) * l**(1-alp) - delta


# utility function 
# I changed this: only if c.le.0, a small amout is added @
def utility(x,y): 
    if s==1:
        return  log(x+psi) + gam*log(y)
    else:
        return (( (x+psi) * (y**gam) )**(1-s) - 1)/ (1-s) 

# Bellman equation of the retired
# input: next-period wealth a'
# output: rhs of Bellman equation
def value1(x):
    c = (1+r)*k0 +  pen - x
    if c<=0:
        return neg
    else:
        return utility(c,1) + b*rvalue(x)
    
# interpolation of next-period value function
# for the retired at wealth k    
def rvalue(k):
    return vr_polate(k)

# Bellman equation of the worker
# input: next-period wealth a'
# output: rhs of Bellman equation

def value2(x):
    # optimal labor supply implied by first-order condition
    n= 1/(1+gam) * (1- gam/((1-tau)*w) * (psi+(1+r)*k0-x) )
    
    if n<0: # corner solution 0<=n<=1
        n = 0
    elif n>=1: # corner solution
        n = 1
    
    c = (1-tau)*w*n + (1+r)*k0 - x
    if c<=0:
        return neg
    
    return utility(c,1-n) + b*vw_polate(x)
 


# searches the MAXIMUM using golden section search
# see also Chapter 11.6.1 in Heer/Maussner, 2009,
# Dynamic General Equilibrium Modeling: Computational
# Methods and Applications, 2nd ed. (or later)
def GoldenSectionMax(f,ay,by,cy,tol):
    r1 = 0.61803399 
    r2 = 1-r1
    x0 = ay
    x3 = cy  
    if abs(cy-by) <= abs(by-ay):
        x1 = by 
        x2 = by + r2 * (cy-by)
    else:
        x2 = by 
        x1 = by - r2 * (by-ay)
    
    f1 = - f(x1)
    f2 = - f(x2)

    while abs(x3-x0) > tol*(abs(x1)+abs(x2)):
        if f2<f1:
            x0 = x1
            x1 = x2
            x2 = r1*x1+r2*x3
            f1 = f2
            f2 = -f(x2)
        else:
            x3 = x2
            x2 = x1
            x1 = r1*x2+r2*x0
            f2 = f1
            f1 = -f(x1)
            
    if f1 <= f2:
        xmin = x1
    else:
        xmin = x2
    
    return xmin

def testfunc(x):
    return -x**2 + 4*x + 6

# test goldensectionsearch
xmax = GoldenSectionMax(testfunc,-2.2,0.0,10.0,0.001)
print(xmax)

# Part 3: Set numerical parameters
# Adjust linear/cubic interpolation in this part


kmin = 0            # inidividual wealth
kmax = 10           # upper limit of capital grid 
na = 200            # number of grid points over assets a in [kmin,kmax]
a =  np.linspace(kmin, kmax, na)   # asset grid 
aeps = (a[1]-a[0])/na     #  test for corner solution
psi = 0.001         # parameter of utility function */
phi = 0.8           # updating of the aggregate capital stock K in outer loop
tol = 0.0001        # percentage deviation of final solution 
tol1 = 1e-10        # tolerance for golden section search 
neg = -1e10         # initial value for value function 
nq = 30             # maximum number of iteration over K
#nq = 1
# Part 4: Start clock for measuring computational time 
start_time = time.time()


# Part 5: Parameterization of the model
# demographics


b = 0.96         # discount factor 
r = 0.045        # initial value of the interest rate 
s = 2            # coefficient of relative risk aversion 
alp = 0.36       # production elasticity of capital 
rep = 0.30       # replacement rate of pensions
delta = 0.10     # rate of depreciation 
tr = 20          # years during retired 
t = 40           # years of working time/
tau = rep / (2+rep)    # equilibrium income tax rate: balanced budget
gam = 2          # disutility from working 
kinit = 0       # capital stock at the beginning of first period

# initialization of aggregate variables
# Step 1 in Algorithm 9.1.1 in Heer/Maussner, DSGE Modeling (Springer) 
nbar = 0.2      # aggregate labor N
kbar = (alp/(r+delta))**(1/(1-alp))*nbar # aggregate capital K
kold = 100      # capital stock in previous iteration of K, initialization
nold = 2        # labor in previous iteration of N, initialization
kq = np.zeros((nq,2))   # saves (K,N) in each outer iteration

# Part 6: iteration of policy function, wealth distribution,..
q = -1
crit = 1+tol
while q<nq-1 and crit>tol:
    q = q+1
    print("q: " + str(q))    
    crit = abs((kbar-kold)/kbar)    # percentage deviation of solution for K
                                    # in iteration q and q-1
    crit0 = abs((nbar-nold)/nbar)
    
    # Step 2 in Algorithm 9.1.1: computation of equilibrium values
    # for w, r, pen
    w = wage_rate(kbar, nbar)       # wage rate
    r = interest_rate(kbar, nbar)   # interest rate
    pen = rep*(1-tau)*w*nbar*3/2    # pensions: balanced social security budget
    kold = kbar
    nold = nbar
    kq[q,0] = kbar
    kq[q,1] = nbar
    
    
    # retired agents' value function  
    vr = np.zeros((na,tr))  # value function 
    aropt = np.zeros((na,tr))  # optimal asset */
    cropt = np.zeros((na,tr))   # optimal consumption */
    for l in range(na):
        vr[l,tr-1] = utility(a[l]*(1+r)+pen,1)
        cropt[l,tr-1] = a[l]*(1+r)+pen
    
    
    # workers' value function 
    vw = np.zeros((na,t))
    awopt = np.ones((na,t))
    cwopt = np.zeros((na,t))
    nwopt = np.zeros((na,t))
    
    
    # compute retiree's policy function 
    for i in  range(tr-1,0,-1):     # all ages i=T+TR,T+TR-1,..,T
        print("q,i,K: " + str([q,t+i,kbar]))
        if _VI_method == 1:
            vr_polate = interpolate.interp1d(a,vr[:,i], fill_value='extrapolate')
        else: 
            vr_polate = interpolate.interp1d(a,vr[:,i],kind='cubic', fill_value='extrapolate')
            
        m0 = 0
             
        for l in range(na): # asset holding at age i
            k0 = a[l]
            # triple ax, bx, cx: [ax,bx] bracket the maximum of Bellman eq.
            ax = 0
            bx = -1
            cx = -2
            v0 = neg
            m = max(-1,m0-2)
            # locate ax <= a' <= cx that bracket the maximum
            while ax>bx or bx>cx:
                m = m+1
                v1 = value1(a[m])
                
                if v1>v0:
                    if m==0: # new value at lower bound a[0]
                        ax=a[m] 
                        bx=a[m]
                    else:
                        bx=a[m] 
                        ax=a[m-1]
                    
                    v0 = v1
                    m0=m   # monotonocity of the value function 
                    
                else:
                    cx=a[m]
                
                if m==na-1:
                    ax = a[m-1]
                    bx = a[m] 
                    cx = a[m] 
                
            
            
            if ax==bx:  # corner solution: a'=0?
                if value1(ax)>value1(aeps):
                    aropt[l,i-1]=0
                else:
                    aropt[l,i-1] = GoldenSectionMax(value1,ax,aeps,cx,tol1)
                
            elif bx==cx:  # corner solution: a'=a[na-1]=kmax?
                if value1(a[na-1])>value1(a[na-1]-aeps):
                    aropt[l,i-1] = a[na-1]
                else:
                     aropt[l,i-1] = GoldenSectionMax(value1,a[na-2],kmax-aeps,kmax,tol1) 
            else:
                aropt[l,i-1] = GoldenSectionMax(value1,ax,bx,cx,tol1)
            

            k1 = aropt[l,i-1]
            vr[l,i-1] = value1(aropt[l,i-1])
            cropt[l,i-1] = (1+r)*a[l]+pen-k1
            
            #print("q,i,l,K" + str([q,i,l,kbar]))
        # print policy function
        if q==0 and i==10:
            
            plt.xlabel('wealth a')
            plt.ylabel('value function of the retiree after 10 years of retirement')
            plt.plot(a,vr[:,i-1])
            plt.show()
            
            
            plt.xlabel('wealth a')
            plt.ylabel('savings of the retiree after 10 years of retirement')
            plt.plot(a,aropt[:,i-1]-a)
            plt.show()
            
            
            plt.xlabel('wealth a')
            plt.ylabel('consumption of the retiree after 10 years of retirement')
            plt.plot(a,cropt[:,i-1])
            plt.show()
            
    # compute worker's policy function 
    for i in  range(t,0,-1):     # all ages i=T,T-1,..1T
        print("q,i,K: " + str([q,i,kbar]))   
        if i==t:
            vw0=vr[:,0]
        else:
            vw0=vw[:,i]
        
        if _VI_method == 1:
            vw_polate = interpolate.interp1d(a,vw0, fill_value='extrapolate')
        else: 
            vw_polate = interpolate.interp1d(a,vw0,kind='cubic', fill_value='extrapolate')
                
        m0 = 0 
        
        for l in range(na): # asset holding at age i
            k0 = a[l]
            # triple ax, bx, cx: [ax,bx] bracket the maximum of Bellman eq.
            ax = 0
            bx = -1
            cx = -2
            v0 = neg
            m = max(-1,m0-2)
            # locate ax <= a' <= cx that bracket the maximum
            while ax>bx or bx>cx:
                m = m+1
                v1 = value2(a[m])
        
                if v1>v0:
                    if m==0: # new value at lower bound a[0]
                        ax=a[m] 
                        bx=a[m]
                    else:
                        bx=a[m] 
                        ax=a[m-1]
                    
                    v0 = v1
                    m0=m   # monotonocity of the value function 
                    
                else:
                    cx=a[m]
                
                if m==na-1:
                    ax = a[m-1]
                    bx = a[m] 
                    cx = a[m] 
        
        
            
            if ax==bx:  # corner solution: a'=0?
                if value2(ax)>value2(aeps):
                    awopt[l,i-1]=0
                else:
                    awopt[l,i-1] = GoldenSectionMax(value2,ax,aeps,cx,tol1)
                
            elif bx==cx:  # corner solution: a'=a[na-1]=kmax?
                if value2(a[na-1])>value2(a[na-1]-aeps):
                    awopt[l,i-1] = a[na-1]
                else:
                     awopt[l,i-1] = GoldenSectionMax(value2,a[na-2],kmax-aeps,kmax,tol1) 
            else:
                awopt[l,i-1] = GoldenSectionMax(value2,ax,bx,cx,tol1)
            

            k1 = awopt[l,i-1]
            
            n = 1/(1+gam)*(1-gam/((1-tau)*w)*(psi+(1+r)*k0-k1))
            
            if n<0:
                n=0
                
            cwopt[l,i-1] = (1-tau)*w*n+(1+r)*k0-k1
            nwopt[l,i-1] = n
            vw[l,i-1] = value2(awopt[l,i-1])
            
            
            #print("q,i,l,K" + str([q,i,l,kbar]))
        if q==0 and i==10:
            
            plt.xlabel('wealth a')
            plt.ylabel('value function of the worker at age 10')
            plt.plot(a,vw[:,i-1])
            plt.show()
            
            
            plt.xlabel('wealth a')
            plt.ylabel('savings of the worker at age 10 year')
            plt.plot(a,awopt[:,i-1]-a)
            plt.show()
            
            
            plt.xlabel('wealth a')
            plt.ylabel('consumption of the worker at age 10')
            plt.plot(a,cwopt[:,i-1])
            plt.show()
        
        
            plt.xlabel('wealth a')
            plt.ylabel('labor supply of the worker at age 10')
            plt.plot(a,nwopt[:,i-1])
            plt.show()
        
    # Part 7: aggregation corresponding to     
    # Step 3-5 of algorithm 9.1.1   
    # computation of the aggregate capital stock and employment nbar 
    kgen = np.zeros(t+tr)
    ngen = np.zeros(t)
    cgen = np.zeros(t+tr)

    kgen[0] = kinit
    
    for j in range(t+tr-1):
        print(j)
        
        if j<t: # worker
       
            if _VI_method == 1:
                aw_polate = interpolate.interp1d(a,awopt[:,j], fill_value='extrapolate')
            else: 
                aw_polate = interpolate.interp1d(a,awopt[:,j],kind='cubic', fill_value='extrapolate') 
            
            kgen[j+1] = aw_polate(kgen[j])
            n = 1/(1+gam)*(1-gam/((1-tau)*w)*(psi+(1+r)*kgen[j]-kgen[j+1]))
            ngen[j] = n
            cgen[j] = (1-tau)*w*n + (1+r)*kgen[j] - kgen[j+1]
        else: # retired
            if _VI_method == 1:
                ar_polate = interpolate.interp1d(a,aropt[:,j-t], fill_value='extrapolate')
            else: 
                ar_polate = interpolate.interp1d(a,aropt[:,j-t],kind='cubic', fill_value='extrapolate') 
            
            kgen[j+1] = ar_polate(kgen[j])
            cgen[j] = (1+r)*kgen[j] + pen - kgen[j+1]
            
    cgen[t+tr-1] = (1+r)*kgen[t+tr-1] + pen
    
    knew = np.mean(kgen)
    
    # update aggregate variables
    kbar = phi*kold + (1-phi)*knew
    nnew = np.mean(ngen)*2/3 # workers have share 2/3 in total population
    nbar = phi*nold + (1-phi)*nnew
   

print("Solution for aggregate capital stock K: " + str(kbar))
print("Solution for aggregate labor N: " + str(nbar))

# plotting the value and policy function

fig, axes = plt.subplots(2, 1, figsize=(8, 16))
axes[0].set_xlabel('age')
axes[0].set_ylabel('wealth a')
axes[0].plot(kgen)
axes[1].set_xlabel('age')
axes[1].set_ylabel('consumption')
axes[1].plot(cgen)
plt.show()                


plt.xlabel('age')
plt.ylabel('labor')
plt.plot(ngen)
plt.show()
        
print("runtime: --- %s seconds ---" % (time.time() - start_time))    
sec = (time.time() - start_time)
ty_res = time.gmtime(sec)
res = time.strftime("%H : %M : %S", ty_res)
print(res)
  
                        
