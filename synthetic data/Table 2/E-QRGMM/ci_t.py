import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad
from sklearn.utils import resample
from tqdm import trange
import statsmodels.formula.api as smf
import os
import ray
import time

import warnings 
warnings.filterwarnings("ignore")

d = 3 # dimensions of covariates
n = 10000 # number of training data

#taulb = 0.1
#tauub = 0.9




x1lb=0
x1ub=10
x2lb=-5
x2ub=5
x3lb=0
x3ub=5
# range of covariates

a0=5
a1=1
a2=2
a3=0.5
r0=1
r1=0.1
r2=0.2
r3=0.05

# coefficients of t Scenario 
dft = 5
# covariates of test data, x0=(1, 4, -1, 3)
x0_1=4
x0_2=-1
x0_3=3



def fit_model(q,mod): # quantile regression
    res = mod.fit(q=q)
    return [q, res.params['Intercept'], res.params['A'],res.params['B'],res.params['C']] 


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
        mask = (x_fine_clipped >= x0) & (x_fine_clipped <= x1)
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




@ray.remote
def boot_once(df,k):
    warnings.filterwarnings('ignore')
    bootstrap_sample = df.sample(n=len(df), replace=True)
    bootstrap_sample = pd.DataFrame(bootstrap_sample, columns=['A','B','C','F'])
    
    mod = smf.quantreg('F ~ A + B + C', bootstrap_sample)
    models = [fit_model(x,mod) for x in quantiles]
    models = pd.DataFrame(models, columns=['q', 'b0', 'b1','b2','b3']) 
    nmodels=models.to_numpy()

    ones_column = np.ones((n, 1))

    X = np.hstack((ones_column,bootstrap_sample[['A','B','C']].values))
    Y = bootstrap_sample['F'].values
    Beta = nmodels[:, 1:]


    delta = 1e-1
    ex = np.mean(X,axis=0)

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
            # print(Lambda)
            derivative_list.append(2 * delta * np.linalg.inv(Lambda) @ ex)
        else:
            derivative_list.append(np.zeros(d+1))
    derivative_list = np.array(derivative_list)
    # print(derivative_list)



    u = np.random.rand(k)
    #Tau = quantiles
    Tau = nmodels.T[:1, :][0]


    beta_curve_cubic = []
    for i in range(d+1):
        beta_inter =  cubic_hermite_interpolation(Tau, Beta.T[i], derivative_list.T[i], u, taulb, tauub)
        beta_curve_cubic.append(beta_inter)

    beta_curve_cubic = np.array(beta_curve_cubic).reshape((d+1,k))

    x0 = np.array([1,x0_1,x0_2,x0_3]).reshape(d+1,1)
    gen_Y =   beta_curve_cubic.T @ x0

    mean = np.mean(gen_Y)
    quantile = np.percentile(gen_Y,80)
    prob = np.sum(gen_Y > quantile_gt)/len(gen_Y)
     
    return [mean,quantile,prob]


taulb = 0.1
tauub = 0.9
m=100
quantiles = create_custom_grid(m,taulb,tauub) # quantile levels in QRGMM
len_q = len(quantiles)



loc = a0+a1*x0_1+a2*x0_2+a3*x0_3
scale = r0+r1*x0_1+r2*x0_2+r3*x0_3
mean_gt = loc
mean_cover_flag = 0


quantile_gt =  stats.t.ppf(0.8,dft,loc,scale)
quantile_cover_flag = 0

prob_gt = 0.20
prob_cover_flag = 0



runi = 100
num_samples = 100
k = 100000

m_length = []
q_length = []
p_length = []
time_list = []

cpu = os.cpu_count() // 2

ray.init(num_cpus = cpu)

with trange(runi,dynamic_ncols=False) as pbar:
    for iter in pbar:
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
            F[i,3]=stats.t.rvs(dft,g1[i],g2[i])
        df = pd.DataFrame(F, columns=['A','B','C','F']) 
        
        start_time = time.time()
        futures = [boot_once.remote(df,k) for _ in range(num_samples)]
        boot_list = ray.get(futures)
        end_time = time.time()


        boot_list_array = np.array(boot_list)
        mean_list = boot_list_array[:, 0]
        quantile_list = boot_list_array[:, 1]
        prob_list = boot_list_array[:, 2]


        mean_q5 = np.percentile(mean_list,5)
        mean_q95 = np.percentile(mean_list,95)
        if mean_q5 < mean_gt < mean_q95:
            mean_cover_flag = mean_cover_flag + 1
        mean_coverage = mean_cover_flag/(iter+1)


        quantile_q5 = np.percentile(quantile_list,5)
        quantile_q95 = np.percentile(quantile_list,95)
        if quantile_q5 < quantile_gt < quantile_q95:
            quantile_cover_flag = quantile_cover_flag + 1
        quantile_coverage = quantile_cover_flag/(iter+1)


        prob_q5 = np.percentile(prob_list,5)
        prob_q95 = np.percentile(prob_list,95)
        if prob_q5 < prob_gt < prob_q95:
            prob_cover_flag = prob_cover_flag + 1
        prob_coverage = prob_cover_flag/(iter+1)

        m_length.append(mean_q95-mean_q5)
        q_length.append(quantile_q95-quantile_q5)
        p_length.append(prob_q95-prob_q5)
        time_list.append(end_time-start_time)


        pbar.set_postfix({"\n m coverage": mean_coverage, "m lb": "{:.4f}".format(mean_q5), "m ub": "{:.4f}".format(mean_q95),
                          "\n q coverage": quantile_coverage, "q lb": "{:.4f}".format(quantile_q5), "q ub":"{:.4f}".format(quantile_q95),
                          "\n p coverage": prob_coverage, "p lb": "{:.4f}".format(prob_q5), "p ub":"{:.4f}".format(prob_q95)})


ray.shutdown()

print("mean length: ",np.mean(m_length),np.std(m_length)/np.sqrt(runi))
print("quantile length: ",np.mean(q_length),np.std(q_length)/np.sqrt(runi))
print("prob length: ",np.mean(p_length),np.std(p_length)/np.sqrt(runi))
print("time: ",np.mean(time_list)/num_samples,np.std(time_list)/np.sqrt(runi)/num_samples)

