##################################################
#          ____  _     ___        ____           #
#         / ___|| |   |_ _|_ __  / ___|          #
#         \___ \| |    | || '_ \| |  _           #
#          ___) | |___ | || | | | |_| |          #
#         |____/|_____|___|_| |_|\____|          #
#                                                #
##################################################

# DESCRIBTION:
#
# Sparse, Likelihood-free INference using Gibbs (SLInG).
#
# For more details, please consult our paper:
#
# https://www.biorxiv.org/content/10.1101/2023.09.15.557764v1
#
# Please cite this paper if you use SLInG in your research.
#
# The Gibbs sampler is based on a Metropolis-Hastings algorithm
# following the suggestions given by Turner and Van Zandt in their paper
# "Hierarchical Approximate Bayesian Computation", while also drawing on
# the paper "The Bayesian Lasso" by Park and Casella. Following their approaches
# allows us to avoid any reference to the likelihood as well as to obtain efficient
# dimension reduction of the parameter space, i.e., to perform model discovery.
# 
# The function gibbs_sampler() is the sparse sampling algorithm, i.e. the heart of SLInG. To run 
# SLInG iteratively, call the function chain() (cf. example_01.py. Calling SLInG in this way
# will also result in a set of standard plots). 
# All possible ABC distance measures are contained in the function distance(). 
# To define a new model, include it in the function model() (this can be a call of an external module). 
# Load the data that you aim to fit by including it (via a function call) in the function 
# observations(). To see how to run SLInG and what the different input variables mean, 
# we refer to the script example_01.py.
#
#============================================================

#import simple_network as sinesa # This import statement requires pyJulia.
import numpy as np
import matplotlib.pyplot as plt
import modellib
from tqdm import tqdm
#from joblib import Parallel, delayed
from sklearn.datasets import load_diabetes
import pygtc # Only required for automated plotting
#from scipy.interpolate import splrep, splev, splprep
import time
#import tensorflow
import pickle
from scipy.spatial import distance as ssdis
from numpy import linalg as nli
import dcor
import Adaptation as ada
import os

#============================================================
# Laplacian probability density

def laplace_density(theta,epsilon):
    f = epsilon/(2.)*np.exp(-abs(theta)*epsilon)
    return f

#============================================================
# Normal probability density

def normal_density(theta,sigma):
    f = 1/(2*np.pi*sigma**2)*np.exp(-(theta)**2/2/sigma**2)
    return f

#============================================================
# Cauchy

def cauchy_density(theta,epsilon):
    f = 1/(np.pi*epsilon*(1+theta**2/epsilon**2))
    return f

#============================================================
# The model in question
# If you introduce a new model, you need to call it here.
# You can define it in modellib.py or in a seperate script.

def model(theta,x_obs,name,yi=[],disttype="Chi2"):

    var_sim = 0.0

    if name == "Diabetes" or name == "LinearD":
        y_sim = modellib.efron(theta,x_obs)
    elif name is "ADAPT":
        lS, lP = ada.PreSen(theta,nolink=noli)
        y_sim = [lS,lP]
    elif name is "SINESA":
        y_sim = sinesa.simulate(theta)

    return [y_sim, var_sim]

#============================================================
# The desired distance measure

def distance(y_sim,y_obs,var_obs,var_sim,disttype="Chi2"):

    if disttype == "RMSD":
    #    IQR = max(y_obs)-min(y_obs)  
        IQR = np.percentile(y_obs,75) - np.percentile(y_obs,25)
        d = np.sqrt(np.sum((np.asarray(y_sim)-np.asarray(y_obs))**2)/len(y_obs))/IQR
    elif disttype == "Chi2":
        # Distance is squared in normal distribution. In the end, it's exactly like defining a likelihood
        # based on the observations.
        d = np.sqrt(np.sum((np.asarray(y_sim)-np.asarray(y_obs))**2/(var_sim+var_obs))/len(y_obs))
    elif disttype == "Pearson":
        d = np.sqrt(np.sum((np.asarray(y_sim)-np.asarray(y_obs))**2/abs(np.asarray(y_obs)))/len(y_obs))
    elif disttype == "JS":
        d = np.sqrt(np.sum(ssdis.jensenshannon(np.asarray(y_sim),np.asarray(y_obs))**2))
    elif disttype == "CC":
        # var_obs contains inv of cov; y_obs contains the mean of observations:
        v = np.transpose(y_sim-y_obs)
        d = np.sqrt(np.sum(np.multiply(v,np.matmul(var_obs,v)))/len(y_obs)/len(y_sim))
    elif disttype == "Energy":
        d = dcor.energy_distance(y_sim, y_obs)
    elif disttype == "SP":
        d = ada.measure(y_sim[0],y_sim[1])
    elif disttype == "Matrix":
        d = sinesa.cov_distance(y_obs,y_sim)
    elif disttype == "MatrixO":
        d = sinesa.cov_distance(y_obs,y_sim,orthogonal=True)

    if d != d:
        d = np.inf
    return d

#============================================================
# Calling different observational data for implemented examples.
# If you create a new model, you need to define the corresponding observations
# to which you compare by including it here.
#
# Note that the observations (i.e., the data that you aim to fit) will automatically be 
# saved in the corresponding folder as a file called observations.npy (and observations_xobs.npy).
# If you set the parameter newobs to False when calling the function chain() and copy an
# existing observation file into your directory, you can perform the inference based on this
# file rather than relying on calling observations(). You hence don't need to define your data
# through the function observations() if you already have a file in the right format. Note that newobs
# is False by default, i.e. SLInG will look for an existing file first.

def observations(theta_obs,name,spls=1.,disttype="Chi2",direc="./"):

    y_int = []
    var_obs = 1.0

    if name == "Diabetes":
        (x_obs, target)  = load_diabetes(return_X_y=True, as_frame=False)
        y_obs = target - np.mean(target)
        print(x_obs)
    elif name == "LinearD":
        x_obs = spls[0]
        y_obs = spls[1]
    elif name == "ADAPT":
        y_obs = [1,1] # Pro forma
        x_obs = []
        global noli
        noli = spls.copy()
    elif name == "SINESA":
        x_obs = []
        y_obs = sinesa.observe(theta_obs,disttype)

    print("Estimated noise level: ", np.sqrt(var_obs) )

    return x_obs, y_obs, y_int, var_obs, theta_obs

#============================================================
# Sparsity-inducing priors

def prior_density(ptype,theta,param):

    if ptype == "Laplace":
        prior = laplace_density(theta,param)
    elif ptype == "Normal":
        prior = normal_density(theta,param)
    elif ptype == "Cauchy":
         prior = cauchy_density(theta,param)

    return prior

#============================================================
# Other priors (uniform priors are so far implemented)

def ParamPrior(truncation,theta_k, k):

    if theta_k < truncation[0][k]:
        P = 0.
    elif theta_k > truncation[1][k]:
        P = 0.
    else:
        P = 1.

    return P

#============================================================
# Main function: Sparse sampling algorithm. The function "chain" calls this iteratively.

def gibbs_sampler(x_obs,y_obs, var_obs, name, direc = "./", truncation = [], sparsity_prior = [], dabc_min = 1.0, dabc_scale = 1000, ptype="Laplace", disttype ="Euclidean", count = 0, burnin = 200, samples = 2000, theta_obs = [10,15,0], dabc = None, step = 1., theta = [], epsilon0 = None,y_int=[],betas = [1.0],ns=30):

#    print("Stepsize for proposal distribution: %f" %(step))
    print("Observations: ", theta_obs)
    J = len(betas)
    K = len(theta_obs)
    print("There are %i parameters." %(K))
    print("Parallel tempering at ", J, " temperatures: ", betas)

    if theta == []:
        theta = np.zeros(K)

    if truncation == []:
        truncation = [-100.*np.ones(len(theta)),100.*np.ones(len(theta))]

    # Allowing for some of the variables not to have any sparsity inducing priors.
    if sparsity_prior == []:
        sparsity_prior = np.ones(len(theta),dtype=bool)

    for k in range(K):
        if theta[k] < truncation[0][k]:
            theta[k] = truncation[0][k]
        elif theta[k] > truncation[1][k]:
            theta[k] = truncation[1][k]

    Theta = np.zeros(K)

    for j in range(J):
        if j > 0:
            theta = np.append(theta,Theta)

    print("Initial parameter values:  ", theta[0:K])

    theta_t = np.vstack((theta,theta))

    if epsilon0 is None:
        epsilon0 = np.ones(len(theta))
    else:
        eps = np.ones(K)
        for j in range(J):
            if j > 0:
                epsilon0 = np.append(epsilon0,eps)

    print("Initial hyperparameter values:", epsilon0[:K])

    EPS = np.vstack((epsilon0,epsilon0))

    for j in range(J):
        y_sim, var_sim = model(theta[j*K:(j+1)*K],x_obs,name,yi=y_int,disttype=disttype)
        d0 = distance(y_sim,y_obs,var_obs,var_sim,disttype=disttype)
        if j == 0:
            y_pre = y_sim.copy()
            var_pre = var_sim
            dist = [d0, d0]
            ACC = [True, True]
        else:
            y_pre = np.column_stack((y_pre,y_sim))
            var_pre = np.column_stack((var_pre,var_sim))
            dist = np.column_stack((dist,[d0,d0]))
            ACC = np.column_stack((ACC,[True,True]))

    y_sim, var_sim = model(theta[:K],x_obs,name,yi=y_int,disttype=disttype)
    d0 = distance(y_sim,y_obs,var_obs,var_sim,disttype=disttype)

    if dabc is None:
        dabc = max(dabc_min,d0/dabc_scale)

    print("Initial distance is %f, while dabc is %f" %(d0, dabc))

    index = np.arange(0,K)

    susw = 0
    sw = 0

    if hasattr(step, "__len__"):
        steps = step.copy()
        steps_t = np.vstack((steps,steps))
    else:
        steps = step*np.ones(len(theta))
        steps_t = np.vstack((steps,steps))

    for i in tqdm(range(samples)):
 
        if J > 1:
            ACC = np.vstack((ACC,[True]*J))
            dist = np.vstack((dist,[1]*J))

        for j in range(J): # Parallel tempering

            np.random.shuffle(index)

            if i < min(100, burnin/2.):
                epsilon = epsilon0.copy()
            else:
                # Sample epsilon from conditional posterior based on previous step

                for k in index:
                    eps = 10**np.linspace(-5,5,1000)
                    de = np.diff(eps)
                    de = np.insert(de,0,eps[0])
                    p = []

                    for e in eps:
                        p.extend([prior_density(ptype,np.asarray(theta[j*K+k]),e)])

                    epsilon[j*K+k] = np.random.choice(eps,p=p*de/np.sum(p*de))

            for k in index:
                # Sample parameter from proposal distribution
                dummy = theta.copy()

                if i > max(100,burnin/2.):
                    dummy_step = steps[k]
                    step = np.std(theta_t[:,j*K+k])
                    if step == 0:
                        step = dummy_step
                else:
                    step = steps[k]

                theta_k = np.random.normal(loc=theta[j*K+k],scale=step)
                while theta_k < truncation[0][k] or theta_k > truncation[1][k]:
                    theta_k = np.random.normal(loc=theta[j*K+k],scale=step)
                dummy[j*K+k] = theta_k

                # If we impose sparsity we must take this into account through the prior
                if sparsity_prior[k]:
                    prior_multiplier = prior_density(ptype,theta[j*K+k],epsilon0[j*K+k])
                else:
                    prior_multiplier = 1.

                # Reverse jump q(theta_old | theta_new). This previously used theta_t[-2],
                # i.e. q(theta_old | theta_older), which made the chain non-Markovian.
                orho = prior_multiplier*normal_density(theta[j*K+k]-theta_k,step)*ParamPrior(truncation,theta[j*K+k],k)

                # Generate Model prediction from new parameters

                if sparsity_prior[k]:
                    prior_multiplier = prior_density(ptype,theta_k,epsilon[j*K+k])
                else:
                    prior_multiplier = 1.

                y_sim, var_sim = model(dummy[j*K:(j+1)*K],x_obs,name,yi=y_int,disttype=disttype)
                rho = prior_multiplier*normal_density(theta_k-theta[j*K+k],step)*ParamPrior(truncation,theta_k,k)

                if J ==  1:
                    y_o = y_pre
                    var_o = var_pre
                else:
                    y_o = y_pre[:,j]
                    var_o = var_pre[:,j]

                rat = (np.exp(-0.5*(distance(y_sim,y_obs,var_obs,var_sim,disttype=disttype)/dabc)**2+0.5*(distance(y_o,y_obs,var_obs,var_o,disttype=disttype)/dabc)**2))**betas[j]

                alpha = min(1,rat*rho/orho)
                jump = np.random.uniform(low=0.0, high=1.0)
                if jump <= alpha:
                    theta[j*K+k] = theta_k
                    steps[j*K+k] = step
                    if J > 1:
                        y_pre[:,j] = y_sim
                        var_pre[:,j] = var_sim
                    else:
                        y_pre = y_sim
                        var_pre = var_sim
                        ACC.extend([True])
                else:
                    if J > 1:
                        ACC[i+2,j] = False
                    else:
                        ACC.extend([False])

            if J > 1:
                dist[-1,j] = distance(y_pre[:,j],y_obs,var_obs,var_pre[:,j],disttype=disttype) 
            else:
                dist.extend([distance(y_pre,y_obs,var_obs,var_pre,disttype=disttype)])

        epsilon0 = epsilon.copy()

        if J > 1:
            U1 = np.random.uniform(low=0.0, high=1.0)
            if U1 <= 1/ns:
                susw += 1 
                js = np.random.choice(np.arange(0,J),size=2,replace=False)
                j0 = js[0]
                j1 = js[1]

                y_0, var_0 = model(theta[j0*K:(j0+1)*K],x_obs,name,yi=y_int,disttype=disttype)
                y_1, var_1 = model(theta[j1*K:(j1+1)*K],x_obs,name,yi=y_int,disttype=disttype)
                nd0 = (distance(y_0,y_obs,var_obs,var_0,disttype=disttype)/dabc)
                nd1 = (distance(y_1,y_obs,var_obs,var_1,disttype=disttype)/dabc)
                r = min(1,np.exp(-0.5*(betas[j0]*nd1**2-betas[j1]*nd0**2)+0.5*(betas[j0]*nd0**2+betas[j1]*nd1**2)))
                rho = 1
                for k in range(K):
                    rho_k = prior_density(ptype,theta[j1*K+k],epsilon[j0*K+k])*prior_density(ptype,theta[j0*K+k],epsilon[j1*K+k])/prior_density(ptype,theta[j1*K+k],epsilon[j1*K+k])/prior_density(ptype,theta[j0*K+k],epsilon[j0*K+k])
                rho = rho_k*rho

                U2 = np.random.uniform(low=0.0, high=1.0)
                if U2 <= r*rho:
                    sw +=1 
                    print("Swapping chain ", j0, " and ", j1, r, rho)
                    theta0 = theta[j0*K:(j0+1)*K].copy()
                    theta1 = theta[j1*K:(j1+1)*K].copy()
                    eps0 = epsilon[j0*K:(j0+1)*K].copy()
                    eps1 = epsilon[j1*K:(j1+1)*K].copy()
                    theta[j1*K:(j1+1)*K] = theta0
                    theta[j0*K:(j0+1)*K] = theta1
                    epsilon[j1*K:(j1+1)*K] = eps0
                    epsilon[j0*K:(j0+1)*K] = eps1

        theta_t = np.vstack((theta_t,theta))
        EPS = np.vstack((EPS,epsilon))
        steps_t = np.vstack((steps_t,steps))

    if J > 1:
        print("Accepted %i of %i swaps." %(sw, susw))

    ext = "%05d" % count

    np.savetxt(direc+name+"_"+disttype+"_"+ext+"_step_"+str(step)+"_dabc_"+str(dabc)+"_scale_"+str(dabc_scale)+"_"+ptype+".txt",np.column_stack((EPS,theta_t,dist)))

    print("Accepted samples (per cent): ", 100*np.sum(ACC)/len(ACC))
    print("Final stepsizes: ", steps)

    if J > 1:
        return theta_t[:,:K], EPS[:, :K], dist[:,0], ACC[:,0], x_obs, y_obs, y_int, dabc, steps
    else:
        return theta_t, EPS, dist, ACC, x_obs, y_obs, y_int, dabc, steps

#============================================================
# Plotting function. Creates a few standard plots that help interpret the output. The function "gibbs_sampler" above also
# saves a handful of (.txt) files with all samples that underlie these plots.
# Note that the plotting requires the installation of pygtc.

def summary_plots(theta_t, EPS, dist, x_obs, y_obs, y_int, var_obs, name, burnin, truths, direc = "./", disttype="Chi2", ptype="Laplace"):

    med = np.median(theta_t[burnin:,:],axis=0)
    sig = np.std(theta_t[burnin:,:],axis=0)
    print("MEDIAN: ", med)
    ym, vm = model(med,x_obs,name,yi=y_int,disttype=disttype)
    dm = distance(ym,y_obs,var_obs,vm,disttype=disttype)
    print("The distance to the median ", dm)
    print("STD: ",sig)
    best = theta_t[np.asarray(dist).argmin(),:]
    print("BEST FIT: ", best)
    print(dist[np.asarray(dist).argmin()])

    np.save(direc+"/summary_results.npy",np.vstack((truths,best,med,sig)))

    plt.figure()
    plt.plot(dist)
    plt.xlabel(r"$\mathrm{Iteration}$",fontsize=13)
    plt.ylabel(r"$\delta$",fontsize=13)
    plt.yscale("log")
    plt.figure()
    if name == "Diabetes":
        plt.plot(x_obs[:,0],y_obs,"co",alpha=0.3)
        plt.plot(x_obs[:,0],model(best,x_obs,name,disttype=disttype)[0],"ro",alpha=0.3)
    else:
        if x_obs == []:
            plt.plot(y_obs,"co")
            plt.plot(model(best,x_obs,name,yi=y_int,disttype=disttype)[0],"r^")
            plt.plot(model(med,x_obs,name,yi=y_int,disttype=disttype)[0],"gs")
            print("Best model output: ", model(best,x_obs,name,yi=y_int,disttype=disttype)[0])
            np.save(direc+"best_output.npy",model(best,x_obs,name,yi=y_int,disttype=disttype)[0])
        else:
            if len(x_obs.shape) == 1:
                plt.plot(x_obs,y_obs,"co")
                plt.plot(x_obs,model(best,x_obs,name,yi=y_int,disttype=disttype)[0],"r",lw=3)
                plt.plot(x_obs,model(med,x_obs,name,yi=y_int,disttype=disttype)[0],"g--",lw=3)
            else:
                plt.plot(x_obs[:,0],y_obs,"co")
                plt.plot(x_obs[:,0],model(best,x_obs,name,disttype=disttype)[0],"r",lw=3)
                plt.plot(x_obs[:,0],model(med,x_obs,name,disttype=disttype)[0],"g--",lw=3)
    plt.xlabel(r"$t$",fontsize=13)
    if y_int != []:
        plt.ylabel(r"$\mathrm{d}y/\mathrm{d}t$",fontsize=13)
    else:
        plt.ylabel(r"$y(t)$",fontsize=13)
    plt.savefig(direc+name+"_"+disttype+"_"+ptype+"_model_vs_obs.png")
    if y_int != []:
        plt.figure()
        plt.plot(x_obs,y_int,"co")
        plt.plot(x_obs,y_best,"r",lw=3)
        plt.plot(x_obs,y_med,"g--",lw=3)
        plt.xlabel(r"$t$",fontsize=13)
        plt.ylabel(r"$y(t)$",fontsize=13)
        plt.savefig(direc+name+"_"+disttype+"_"+ptype+"_INT_model_vs_obs.png")
    plt.figure()
    colours = ["b","r","g","c","m","y","k","orange","purple","grey","pink","sienna","coral","steelblue","navy","darkgreen"]
    for i in range(len(med)):
        plt.plot(EPS[:,i],color=colours[i])
    plt.axvline(x=burnin,color="grey",linestyle="--")
    plt.xlabel(r"$\mathrm{Iteration}$",fontsize=13)
    plt.ylabel(r"$\epsilon$",fontsize=13)
    plt.yscale("log")
    plt.savefig(direc+name+"_"+disttype+"_"+ptype+"_eps.png")

    plt.figure()
    for i in range(len(med)):
        plt.plot(theta_t[:,i],color=colours[i])
    plt.xlabel(r"$\mathrm{Iteration}$",fontsize=13)
    plt.ylabel(r"$\theta$",fontsize=13)
    plt.axvline(x=burnin,color="grey",linestyle="--")
    plt.savefig(direc+name+"_"+disttype+"_"+ptype+"_params.png")

    plt.figure()
    for i in range(len(med)):
        plt.hist(theta_t[burnin:,i],50,color=colours[i],alpha=0.4)
        plt.axvline(x=truths[i],color=colours[i],linestyle="--",alpha=0.4)
    plt.xlabel(r"$\theta$",fontsize=13)
    plt.ylim(0,200)
    plt.savefig(direc+name+"_hist.png")

    GTC = pygtc.plotGTC(chains=[theta_t[burnin:,:]],
    legendMarker='All',
    figureSize='MNRAS_page',
    plotName=direc+name+"_"+disttype+"_"+ptype+'_contour.pdf'
    )

#    plt.show()

#============================================================
# Function calling sampler iteratively as well as plotting function.
# The reason for calling the sampler in this manner is merely to iteratively adjust dabc_min to the value set by the user. 

def chain(name, theta_obs, truncation, direc="./", sparsity_prior = [] ,dabc_min =1.0, ptype="Laplace", theta_int = [], dabc_scale = 15., count_start =0, disttype = "Euclidean", burnin = 500, samples = 2000, samples_scouts = 1000, stepsize = [1.,1.,1.],spls=1.,newobs=False):

    print("Step series: ", stepsize)

    if newobs == False:
        ofile = direc + "/observations.npy"
        if os.path.exists(ofile):
            print("Loading observations")
            xfile = direc+"/observations_xobs.npy"
            if os.path.exists(xfile):
                x_obs = np.load(xfile)
            else:
                x_obs = []
            y_int = []
            y_obs = np.load(direc+"/observations.npy")
            var_obs = 1.0
        else:
            print("NO OBSERVATION FILE FOUND. Setting newobs to True.")
            newobs = True

    if newobs == True:
        x_obs, y_obs, y_int, var_obs, theta_obs = observations(theta_obs,name,spls=spls,disttype=disttype,direc=direc)

        np.save(direc+"/observations.npy",y_obs)
        if x_obs != []:
            np.save(direc+"/observations_xobs.npy",x_obs)

        print("Saved observations...")

#    print("OBS:", y_obs)

    t_start = time.time()

    if len(stepsize) >= 3:

        print("Running multiple chains to adjust dabc_min...")

        # Only the first entry of stepsize matters. The remaining entries just communicate how many chains we have.

        theta_t, EPS, dist, ACC, x_obs, y_obs, y_int, dabc, steps = gibbs_sampler(x_obs, y_obs,var_obs, name, direc=direc, truncation=truncation, sparsity_prior=sparsity_prior, dabc_min=dabc_min, ptype=ptype, dabc_scale = dabc_scale, count=count_start, burnin=burnin, disttype = disttype, samples=2*samples_scouts, theta_obs=theta_obs, step=stepsize[0],dabc=None,theta=theta_int,y_int=y_int)
        best = np.asarray(dist).argmin()
        theta_int = theta_t[best,:]
        epsilon0 = EPS[min(best+1,len(theta_t)-1),:]
        for j, step in enumerate(stepsize[1:-1]):
            theta_t, EPS, dist, ACC, x_obs, y_obs, y_int, dabc, steps = gibbs_sampler(x_obs, y_obs,var_obs, name, direc=direc, truncation=truncation, sparsity_prior=sparsity_prior, burnin=1, dabc_min=dabc_min, ptype=ptype, dabc_scale = dabc_scale, disttype = disttype, samples=samples_scouts, theta_obs=theta_obs, step=steps,dabc=None,theta=theta_int,epsilon0=epsilon0, count=count_start+j+1,y_int=y_int)
            best = np.asarray(dist).argmin()
            theta_int = theta_t[best,:]
            epsilon0 = EPS[min(best+1,len(theta_t)-1),:]

        theta_t, EPS, dist, ACC, x_obs, y_obs, y_int, dabc, steps = gibbs_sampler(x_obs, y_obs,var_obs, name, direc=direc, truncation=truncation, sparsity_prior=sparsity_prior, burnin=1, dabc_min=dabc_min, ptype = ptype, disttype = disttype, dabc_scale = dabc_scale, samples=samples, theta_obs=theta_obs, step=steps,dabc=None,theta=theta_int,epsilon0=epsilon0,count=count_start+j+2,y_int=y_int)

    elif len(stepsize) == 1:

        print("Running a single chain...")

        theta_t, EPS, dist, ACC, x_obs, y_obs, y_int, dabc, steps = gibbs_sampler(x_obs, y_obs,var_obs, name, direc=direc, truncation=truncation, sparsity_prior=sparsity_prior, dabc_min=dabc_min, ptype=ptype, dabc_scale = dabc_scale, count=count_start, burnin=burnin, disttype = disttype, samples=samples, theta_obs=theta_obs, step=stepsize[0],dabc=None,theta=theta_int,y_int=y_int)

    else:
        print("Either use one stepsize or at least 3 different ones. Run terminated.")
        return 0


    t_end = time.time()

    print("Duration of chain %i s" %(t_end-t_start))

    summary_plots(theta_t, EPS, dist, x_obs, y_obs, y_int, var_obs, name, burnin, theta_obs, direc=direc, disttype=disttype, ptype=ptype)

    return theta_t[np.asarray(dist).argmin(),:]
