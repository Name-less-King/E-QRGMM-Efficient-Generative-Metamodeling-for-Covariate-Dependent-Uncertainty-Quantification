import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import wasserstein_distance

import warnings 
warnings.filterwarnings("ignore")

d = 3 # dimensions of covariates
n = 10000 # number of training data

# range of covariates

x1lb=0
x1ub=10
x2lb=-5
x2ub=5
x3lb=0
x3ub=5


# coefficients of normal distribution

a0=5
a1=1
a2=2
a3=0.5
a_array = [a0,a1,a2,a3]

r0=1
r1=0.1
r2=0.2
r3=0.05
r_array = [r0,r1,r2,r3]



############################### generate data ###############################
    
u1=np.random.rand(n)
x1=x1lb+(x1ub-x1lb)*u1
u2=np.random.rand(n)
x2=x2lb+(x2ub-x2lb)*u2
u3=np.random.rand(n)
x3=x3lb+(x3ub-x3lb)*u3

g1=a0+a1*x1+a2*x2+a3*x3
g2=r0+r1*x1+r2*x2+r3*x3
F=np.zeros((n,4))
for i in np.arange(0,n):
    F[i,0]=x1[i]
    F[i,1]=x2[i]
    F[i,2]=x3[i]
    F[i,3]=np.random.normal(g1[i],g2[i])
df = pd.DataFrame(F, columns=list('A''B''C''F')) 


def fit_model(q,mod): # quantile regression
    res = mod.fit(q=q)
    return [q, res.params['Intercept'], res.params['A'],res.params['B'],res.params['C']] 
def normfun(x, mu, sigma): # pdf of the normal distribution 
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf
def g1fun(x0_1,x0_2,x0_3): # mean function of Scenario 1
    g1=a0+a1*x0_1+a2*x0_2+a3*x0_3
    return g1 
def g2fun(x0_1,x0_2,x0_3): # standard deviation function of Scenario 1
    g2=r0+r1*x0_1+r2*x0_2+r3*x0_3
    return g2 

# Function for cubic Hermite interpolation
def cubic_hermite_interpolation(x_data, y_data, dy_data, x_fine,lb,ub):
    # Calculate the coefficients of the cubic polynomials
    n = len(x_data)
    y_fine = np.zeros_like(x_fine)
    
    x_fine_clipped = np.clip(x_fine, lb, ub)
    
    for i in range(n-1):
        # Get the segment between x_data[i] and x_data[i+1]
        x0, x1 = x_data[i], x_data[i+1]
        y0, y1 = y_data[i], y_data[i+1]
        dy0, dy1 = dy_data[i], dy_data[i+1]
        
        # Calculate the cubic Hermite basis functions
        h = x1 - x0 + 1e-8
        t = (x_fine - x0) / h
        
        # Only compute for x_fine within the range of x0 and x1
        mask = (x_fine_clipped > x0) & (x_fine_clipped < x1)
        t_masked = (x_fine_clipped[mask] - x0) / h
        
        # Compute the cubic Hermite basis functions for the masked values
        h00 = (2 * t_masked**3 - 3 * t_masked**2 + 1)
        h10 = (t_masked**3 - 2 * t_masked**2 + t_masked)
        h01 = (-2 * t_masked**3 + 3 * t_masked**2)
        h11 = (t_masked**3 - t_masked**2)
        
        # Compute the interpolated values for the current segment
        y_fine[mask] = h00 * y0 + h10 * h * dy0 + h01 * y1 + h11 * h * dy1
        
    out_of_bounds_mask = (x_fine < lb) | (x_fine > ub)
    y_fine[out_of_bounds_mask] = np.interp(x_fine[out_of_bounds_mask], x_data, y_data)
        
    return y_fine


def create_custom_grid(m, taulb, tauub, c=2):
    """
    Create custom grid points:  
    - Uniformly distribute m-1 points on (1/m, 1-1/m) (using linspace)  
    - Insert points within (taulb, tauub) at intervals of (1/m)^0.4, multiply the number of inserted points by a factor of c, and distribute them uniformly
    """
    base_points = np.linspace(1/m, 1 - 1/m, m - 1)  
    # find points in (taulb, tauub) 
    mask = (base_points > taulb) & (base_points < tauub)

    spacing = (1/m) ** 0.4
    num_points = int((tauub - taulb) / spacing) * c  
    if num_points > 0:
        new_inner_points = np.linspace(taulb, tauub, num_points + 2)[1:-1]  
    else:
        new_inner_points = np.array([])

    outer_points = base_points[~mask]
    combined_points = np.concatenate([outer_points, new_inner_points, [taulb, tauub]])
    combined_points = np.unique(np.round(combined_points, decimals=6))

    return combined_points


# covariates of test data, x0=(1, 4,-1,3)
x0_1=4
x0_2=-1
x0_3=3



xl=2
xr=15
X_pdf = np.arange(xl,xr,0.01)
Y_pdf = normfun(X_pdf, g1fun(x0_1,x0_2,x0_3),g2fun(x0_1,x0_2,x0_3))  


def QRGMM_naive(m):
    le = 1/m
    ue = 1 - 1/m

    quantiles = np.linspace(le,ue, m-1, endpoint=True) # quantile levels in QRGMM
    len_q = len(quantiles)
    ################################ QRGMM ###################################

    mod = smf.quantreg('F ~ A + B + C', df)
    models = [fit_model(x,mod) for x in quantiles]
    models = pd.DataFrame(models, columns=['q', 'b0', 'b1','b2','b3']) 
    nmodels=models.to_numpy()

 

    Beta = nmodels[:, 1:]

    k = 100000
    u = np.random.rand(k)
    Tau = nmodels.T[:1, :][0]


    beta_curve_linear = []
    for i in range(d+1):
        beta_inter = np.interp(u,Tau, Beta.T[i])
        beta_curve_linear.append(beta_inter)

    beta_curve_linear = np.array(beta_curve_linear).reshape((d+1,k))


    x0 = np.array([1,x0_1,x0_2,x0_3]).reshape(d+1,1)


    gen_Y_linear = beta_curve_linear.T @ x0
    gen_Y_linear = gen_Y_linear.flatten()
    D_linear, _ = stats.kstest(gen_Y_linear, 'norm',args=(g1fun(x0_1,x0_2,x0_3),g2fun(x0_1,x0_2,x0_3)))
    Y = F[:, -1]
    WD_linear = wasserstein_distance(gen_Y_linear,Y.flatten())
    return [D_linear,WD_linear,len_q]



def E_QRGMM(m,taulb,tauub):

    quantiles = create_custom_grid(m,taulb,tauub) # quantile levels in QRGMM
    len_q = len(quantiles)
    ################################ E-QRGMM ###################################

    mod = smf.quantreg('F ~ A + B + C', df)
    models = [fit_model(x,mod) for x in quantiles]
    models = pd.DataFrame(models, columns=['q', 'b0', 'b1','b2','b3']) 
    nmodels=models.to_numpy()

    ones_column = np.ones((n, 1))

    X = np.hstack((ones_column,F[:, :3]))
    Y = F[:, -1]
    Beta = nmodels[:, 1:]


    delta = 1e-1
    ex = np.array([1,np.mean(x1),np.mean(x2),np.mean(x3)])

    derivative_list = []
    for j in range(Beta.shape[0]):
        if nmodels[j,0] > (taulb-1e-6) and nmodels[j,0] < (tauub+1e-6):
            beta = Beta[j].reshape(Beta[j].shape[0],1)
            expectation_list = []
            for i in range(n):
                x = X[i].reshape(X[i].shape[0],1)
                y = Y[i]
                indicator = np.abs(y - x.T @ beta)

                if  indicator < delta:
                    expectation_list.append(x @ x.T)
                else :
                    expectation_list.append(np.zeros((d+1,d+1)))

            Lambda = np.mean(expectation_list,axis=0)

            derivative_list.append(2 * delta * np.linalg.inv(Lambda) @ ex)
        else:
            derivative_list.append(np.zeros(d+1))
    derivative_list = np.array(derivative_list)


    k = 100000
    u = np.random.rand(k)
    Tau = nmodels.T[:1, :][0]


    beta_curve_linear = []
    for i in range(d+1):
        beta_inter = np.interp(u,Tau, Beta.T[i])
        beta_curve_linear.append(beta_inter)

    beta_curve_linear = np.array(beta_curve_linear).reshape((d+1,k))


    beta_curve_cubic = []
    for i in range(d+1):
        beta_inter =  cubic_hermite_interpolation(Tau, Beta.T[i], derivative_list.T[i], u, taulb, tauub)
        beta_curve_cubic.append(beta_inter)

    beta_curve_cubic = np.array(beta_curve_cubic).reshape((d+1,k))

    x0 = np.array([1,x0_1,x0_2,x0_3]).reshape(d+1,1)

    gen_Y_cubic =   beta_curve_cubic.T @ x0
   
    gen_Y_cubic = gen_Y_cubic.flatten()
    D_cubic, _ =stats.kstest(gen_Y_cubic, 'norm',args=(g1fun(x0_1,x0_2,x0_3),g2fun(x0_1,x0_2,x0_3)))

    WD_cubic = wasserstein_distance(gen_Y_cubic,Y.flatten())

    
    return [D_cubic,WD_cubic,len_q]


runi = 100
m_list = [12,30,50,70,100,130,160,200,250,300,400,500,600,700]
D_cubic_list = []
WD_cubic_list = []

custom_lenq = []
taulb = 0.1
tauub = 0.9
for m in m_list:
    custom_lenq.append(len(create_custom_grid(m,taulb,tauub)))



# Perform the computation runi times and take the average as the result
D_cubic_avg_list = []
WD_cubic_avg_list = []


for m in m_list:
    D_cubic_sum = 0
    WD_cubic_sum = 0
    for _ in range(runi):
        D_cubic, WD_cubic, _ = E_QRGMM(m, taulb, tauub)
        D_cubic_sum += D_cubic
        WD_cubic_sum += WD_cubic
    D_cubic_avg_list.append(D_cubic_sum / runi)
    WD_cubic_avg_list.append(WD_cubic_sum / runi)

D_cubic_list = D_cubic_avg_list
WD_cubic_list = WD_cubic_avg_list


D_baseline_list = []
WD_baseline_list = []
baseline_len = []



# Perform the computation runi times and take the average as the result

D_baseline_avg_list = []
WD_baseline_avg_list = []

for m in custom_lenq:
    D_baseline_sum = 0
    WD_baseline_sum = 0
    for _ in range(runi):
        D_linear, WD_linear, _ = QRGMM_naive(m+1)
        D_baseline_sum += D_linear
        WD_baseline_sum += WD_linear
    D_baseline_avg_list.append(D_baseline_sum / runi)
    WD_baseline_avg_list.append(WD_baseline_sum / runi)

D_baseline_list = D_baseline_avg_list
WD_baseline_list = WD_baseline_avg_list



# Plot KS statistics for E-QRGMM and QRGMM
plt.figure(figsize=(8, 6))
plt.plot(custom_lenq, D_cubic_list, label='E-QRGMM', marker='o', linestyle='-', color='blue')
plt.plot(custom_lenq, D_baseline_list, label='QRGMM', marker='s', linestyle='--', color='green')

# Add labels, title, and legend
plt.xlabel('Total Points')
plt.ylabel('KS')
plt.title('Comparison of Approximation Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.show()