library(SLOPE)

# need to specify p,delta as global variables

################################## state evolution equation
F= function(tau,alpha,prior=x0,iteration=100,sigma=0){
  result=0
  for (i in 1:iteration){
    result=result+mean((prox_sorted_L1(prior+tau*rnorm(p),alpha*tau)-prior)^2)/iteration
  }
  return(sigma^2+result/delta)
}


################################## alpha to tau calibration
alpha_to_tau=function(alpha_seq,prior=x0,sigma=0,max_iter=100){
  tau=sqrt(sigma^2+mean(prior^2)/delta)
  record_tau=rep(0,max_iter) # initialize

  for (t in 1:max_iter){
    tau=sqrt(F(tau,alpha_seq,prior,sigma=sigma))
    record_tau[t]=tau #record each tau
  }
  return(mean(record_tau))
}

################################## alpha to lambda calibration
alpha_to_lambda=function(alpha_seq,max_iter=100,prior=x0,second_iter=100,sigma=0){
  tau=alpha_to_tau(alpha_seq,prior,sigma,max_iter)

  E=0
  for (t in 1:second_iter){
    prox_solution=prox_sorted_L1(prior+tau*rnorm(p),alpha_seq*tau)
    E=E+length(unique(abs(prox_solution)))/p/delta/second_iter
  }
  lambda_seq=(1-E)*alpha_seq*tau

  return(lambda_seq)
}
################################## lambda to alpha calibration (implicitly)
lambda_to_alpha=function(lambda_seq,tol=1e-5,prior=x0,sigma=0){
  # standardize lambda sequence
  lambda_seq=sort(lambda_seq,T)
  l=lambda_seq/lambda_seq[1]


  ### find interval that has larger and smaller first entry at endpoints compared to lambda_seq[1] for bisection
  # starting guess
  alpha1=l/2
  alpha2=l*2

  lambda1=alpha_to_lambda(alpha1,sigma=sigma)
  lambda2=alpha_to_lambda(alpha2,sigma=sigma)

  while ((lambda1[1]<lambda_seq[1])*(lambda2[1]>lambda_seq[1])==0){
    if (lambda1[1]<lambda_seq[1]){
      alpha1=alpha1*2
      alpha2=alpha2*2
    } else{
      alpha1=alpha1/2
      alpha2=alpha2/2
    }
    lambda1=alpha_to_lambda(alpha1,sigma=sigma)
    lambda2=alpha_to_lambda(alpha2,sigma=sigma)
  }


  ### bisection to find the alpha_seq which is parallel to lambda_seq
  while ((alpha2[1]-alpha1[1])>tol){
    middle_alpha_seq=(alpha1+alpha2)/2
    middle_lambda=alpha_to_lambda(middle_alpha_seq,sigma=sigma)
    if (middle_lambda[1]>lambda_seq[1]){
      alpha2=middle_alpha_seq
    }else if (middle_lambda[1]<lambda_seq[1]){
      alpha1=middle_alpha_seq
    }else{
      break
    }
  }
  return(middle_alpha_seq)
}
