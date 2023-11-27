import numpy as np


def sgd(x, etha, w0, iterate, error, gradiantcost):
	"""
	Purpose: Find the optimal with stochastique gradiant descents method
	The algorithm : 
	1 initialization of the weigh
	2 while the convergence criterium not met do
		select a random point if the train data
		compute the gradiant of the cost function at this point
		update the weight with the learning rate
	Input
			x : Traning data
			etha : learning rate
			w0 : weight
	output :
		w : weight
	"""
	wp = w0
	w = wp
	find = False
	for i in range(iterate):
		j = np.random.randint(0, x.shape[0]-1)
		g = gradiantcost(x[j])
		w = wp - etha*g
		er = w-wp
		if np.sum(er) < error:
			find = True
			break
		wp = w
	return w
# end def

def adagrad(x, etha, w0, epsilon, r0, iterate, error, gradiantcost):
	"""
	Purpose: adaptative gradiant descente algorithm, the purpose of the algorithme is to find the
	optimal by ajusting the paramater based on its histry.
	Algrithm
	Input
			x : Traning data
			etha : learning rate
			w0 : weight
			epsilon : fuzz factor
			r : learning rate decay over each update
	output :
		w : weight
	"""
	wp, w, r= w0, wp, r0
	find = False
	for i in range(iterate):
		j = np.random.randint(0, x.shape[0]-1)
		g = gradiantcost(x[j])
		r = r0 + g * g 
		w = wp - (etha/(epsilon + np.sqrt(r)))*g
		er = w-wp
		if np.sum(er) < error:
			find = True
			break
		wp = w
		r0 = r
	return w
# end def

def adadelta(x, etha, w0, epsilon,r10,r20,beta10, beta20, iterate, error, gradiantcost):
	"""
	Purpose: 
	Input : 
	Training data S
	learning rate η
	weights w
	decay rate ρ
	fuzz factor ε
	"""
	wp,w,r1,r2,beta1= w0, wp, r10,r20,beta10
	find = False
	for i in range(iterate):
		# randomly select an integer j
		j = np.random.randint(0, x.shape[0]-1)
		# compute the gradiant of the cost fonction a xj
		g = gradiantcost(x[j])
		# moment updating
		r1 = beta1*r10 + (1 - beta1)*(g*g)
		m = -np.sqrt(r2+epsilon)/(np.sqrt(r1+epsilon))
		# parameter updating
		w = wp + m
		er = w-wp
		r2 = beta1*r2 + (1-beta1)*r2
		if np.sum(er) < error:
			find = True
			break
		wp, r10, r20= w, r1, r2
	return w
# end def

def adam(x, etha, w0, epsilon,r10,r20,beta10, beta20, iterate, error, gradiantcost):
	"""
	Purpose: The algorithm is a variant of the grandiant descent algorithm, the idea is to computer 
	the parameter by utilize two additionnal computation. The mean of the pass computed grandant and 
	the exponential moving average of the gradiant square
	Input :
	 Training data S 
	 learning rate η
	 weights w 
	 fuzz factor ε,
	 learning rates decay over each update r1 and r2
	 exponential decay rates β1 and β2
	"""
	wp,w,r1, r2, beta1, beta2, t= w0, wp, r10, r20, beta10, beta20, 0
	find = False
	for i in range(iterate):
		j = np.random.randint(0, x.shape[0]-1)
		g = gradiantcost(x[j])
		t = t+1
		r1, r2 = beta1*r10 + (1 - beta1)*g, beta2*r20 + (1 - beta1)*(g*g)
		r_, r__= r1/(1-beta1**t), r2/(1-beta2**2)
		w = wp - (etha*r_/(epsilon + np.sqrt(r__)))*g
		er = w-wp
		if np.sum(er) < error:
			find = True
			break
		wp, r10, r20 = w, r1, r2
	return w
# end def

def nadam(x, etha, w0, epsilon,r10,r20,beta10, beta20, iterate, error, gradiantcost):
	"""
	Nesterov-accelerated Adaptive Moment Estimation
	Purpose: 
	"""
	wp,w,r1, r2, beta1, beta2, t= w0, wp, r10, r20, beta10, beta20, 0
	find = False
	for i in range(iterate):
		# randomly select an integer j
		j = np.random.randint(0, x.shape[0]-1)
		# compute the gradiant of the cost fonction a xj
		g = gradiantcost(x[j])
		t = t+1
		# moment updating
		r1, r2 = beta1*r10 + (1 - beta1)*g, beta2*r20 + (1 - beta1)*(g*g)
		# bias correction
		r_, r__= r1/(1-beta1**t), r2/(1-beta2**2)
		# compute  Nesterov Accelerated Gradient
		m = (1-beta1)*r1/(1-beta1**t) + beta1*g
		# parameter updation with the Nesterov Accelerated Gradient
		u = - (etha/(np.sqrt(r__)+epsilon))*m
		# parameter updating
		w = wp + u
		er = w-wp
		if np.sum(er) < error:
			find = True
			break
		wp, r10, r20 = w, r1, r2
	return w
# end def

def adamax(x, etha, w0, epsilon,r10,r20,beta10, beta20, iterate, error, gradiantcost):
	"""
	Purpose: 
	"""
	wp,w,r1, r2, beta1, beta2, t= w0, wp, r10, r20, beta10, beta20, 0
	find = False
	for i in range(iterate):
		# randomly select an integer j
		j = np.random.randint(0, x.shape[0]-1)
		# compute the gradiant of the cost fonction a xj
		g = gradiantcost(x[j])
		t = t+1
		# moment updating
		r1, r2 = beta1*r10 + (1 - beta1)*g,np.maximum(beta2*r20, np.abs(g))
		m = r1/(1-beta1**t)
		u = - (etha/(1-beta1**t))*(m/r2+epsilon)
		# parameter updating
		w = wp + u
		er = w-wp
		if np.sum(er) < error:
			find = True
			break
		wp, r10, r20 = w, r1, r2
	return w
# end def


def RSMprop(x, etha, w0, beta10, epsilon,r10, iterate, error, gradiantcost):
	"""
	Purpose: 
	"""
	wp,w,r1,beta1= w0, wp, r10,beta10
	find = False
	for i in range(iterate):
		# randomly select an integer j
		j = np.random.randint(0, x.shape[0]-1)
		# compute the gradiant of the cost fonction a xj
		g = gradiantcost(x[j])
		# moment updating
		r1 = beta1*r10 + (1 - beta1)*(g*g)
		m = -etha/(np.sqrt(r1) + epsilon)
		# parameter updating
		w = wp + m
		er = w-wp
		if np.sum(er) < error:
			find = True
			break
		wp, r1 = w, r1
	return w
# end def