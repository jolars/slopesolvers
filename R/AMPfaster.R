# source('R/lambda_to_alpha.R')
# library(SLOPE)
# library(withr)
#
# # rewrite SLOPE_solver_iter function in 'SLOPE' package to store each iterates in FISTA
# SLOPE_solver_iter=function (A, b, lambda, initial = NULL, prox = prox_sorted_L1,
#                             max_iter = 10000, grad_iter = 20, opt_iter = 1, tol_infeas = 1e-06,
#                             tol_rel_gap = 1e-06)
# {
#   n = ncol(A)
#   x = with_seed(0,rnorm(n))
#   x = x/sqrt(sum(x^2))
#   x = t(A) %*% (A %*% x)
#   L = sqrt(sum(x^2))
#   x.init = if (is.null(initial))
#     rep(0, n)
#   else initial
#   t = 1
#   eta = 2
#   x = x.init
#   y = matrix(0,max_iter,length(x))
#   y[1,] = x
#   Ax = A %*% x
#   f.prev = Inf
#   iter = 0
#   optimal = FALSE
#   row_index=1
#   time=rep(0,max_iter)
#   repeat {
#     if ((iter%%grad_iter) == 0)
#       r = (A %*% y[row_index,]) - b
#     else r = (Ax + ((t.prev - 1)/t) * (Ax - Ax.prev)) - b
#     g = t(A) %*% r
#     f = as.double(crossprod(r))/2
#     iter = iter + 1
#     if ((iter%%opt_iter) == 0) {
#       gs = sort(abs(g), decreasing = TRUE)
#       ys = sort(abs(y[row_index,]), decreasing = TRUE)
#       infeas = max(max(cumsum(gs - lambda)), 0)
#       obj_primal = f + as.double(crossprod(lambda, ys))
#       obj_dual = -f - as.double(crossprod(r, b))
#       if ((abs(obj_primal - obj_dual)/max(1, obj_primal) <
#            tol_rel_gap) && (infeas < tol_infeas * lambda[[1]]))
#         optimal = TRUE
#     }
#     if (optimal || (iter >= max_iter))
#       break
#     Ax.prev = Ax
#     x.prev = x
#     f.prev = f
#     t.prev = t
#     repeat {
#       x = prox(y[row_index,] - (1/L) * g, lambda/L)
#       d = x - y[row_index,]
#       Ax = A %*% x
#       r = Ax - b
#       f = as.double(crossprod(r))/2
#       q = f.prev + as.double(crossprod(d, g)) + (L/2) *
#         as.double(crossprod(d))
#       if (q < f * (1 - 1e-12))
#         L = L * eta
#       else break
#     }
#     t <- (1 + sqrt(1 + 4 * t^2))/2
#     row_index=row_index+1
#     y[row_index,] <- x + ((t.prev - 1)/t) * (x - x.prev)
#   }
#   if (!optimal)
#     warning("SLOPE solver reached iteration limit")
#   structure(list(x = y, optimal = optimal, iter = iter,
#                  infeas = infeas, obj_primal = obj_primal, obj_dual = obj_dual,
#                  lipschitz = L,times=time), class = "SLOPE_solver.result")
# }
#
#
#
# ############ simulation for MSE with true minimizer
#
# super=Sys.time() # roughly 90 s
# AMP_iter=300
# FISTA_iter=2000
# ISTA_iter=20000
#
# set.seed(3)
# ### setup: iid Gaussian design matrix with 500 rows and 1000 columns
# # prior is Bernoulli-Gaussian with 0.1 probability being standard normal
# # no noise; alpha is Bernoulli(0.5)
# p=1000;delta=0.5;eps=0.1;sigma=0;n=p*delta
# A=matrix(rnorm(n*p,mean=0,sd=1/sqrt(p*delta)), n,p)
# alpha=c(rep(1,p*0.5),rep(0,p*0.5))
#
#
# record_AMP=rep(0,AMP_iter)
# record_FISTA=rep(0,FISTA_iter)
# record_ISTA=rep(0,ISTA_iter)
#
# x0=rnorm(p)*rbinom(p,1,eps)
# y=A%*%x0+sigma*rnorm(n)
# lambda_seq=alpha_to_lambda(alpha,max_iter = 100,second_iter = 1000,sigma = sigma)
# alpha <- lambda_to_alpha(lambda, 1e-6, x0, sigma)
#
# # calculate FISTA iterates and minimizer
# SOLVER=SLOPE_solver_iter(A,y,lambda_seq,tol_infeas = 1e-8,tol_rel_gap = 1e-8,max_iter = 10000)
# slope=SOLVER$x[1:SOLVER$iter,]
# minimizer=slope[nrow(slope),]
#
# # FISTA MSE compared to true minimizer
# record_FISTA[1]=mean((rep(0,p)-minimizer)^2)
# for (i in 2:FISTA_iter){
#   record_FISTA[i]=mean((slope[i,]-minimizer)^2)
# }
#
# ### calculate AMP MSE
# b=rep(0,p);z=y;tau=sqrt(sigma^2+4*eps/delta)
#
# for (i in 1:AMP_iter){
#   # record_AMP[i]=mean((b-minimizer)^2)
#   b=prox_sorted_L1(t(A)%*%z+b,alpha*tau)
#   z=y-A%*%b+z*length(unique(abs(b)))/n
#   tau=sqrt(F(tau,alpha,sigma = sigma,iteration = 1))
# }
#
# # calculate ISTA MSE
#
# t=2/norm(A,'2')^2;b=rep(0,p)
# for (i in 1:ISTA_iter){
#   # record_ISTA[i]=mean((b-minimizer)^2)
#   b=prox_sorted_L1(b+t*t(A)%*%(y-A%*%b),t*lambda_seq)
# }
#
#
# ####### hitting time
# for (t in 2:6){
#   print(paste0('Hitting time at MSE=1e-',t))
#   print(min(which(record_AMP<10^(-t))))
#   print(min(which(record_FISTA<10^(-t))))
#   print(min(which(record_ISTA<10^(-t))))
# }
#
#
#
#
#
#
#
# ############ simulation for set difference with support of true minimizer
#
# symdiff <- function( x, y) { setdiff( union(x, y), intersect(x, y))}
# special_iter=300
#
# set.seed(1)
# ### setup similar to previous simulation
# p=1000;delta=0.5;eps=0.1;sigma=0;n=p*delta
# A=matrix(rnorm(n*p,mean=0,sd=1/sqrt(p*delta)), n,p)
# alpha=c(rep(2,p*0.5),rep(0,p*0.5))
#
#
# sdiff_AMP=rep(0,special_iter)
# sdiff_FISTA=rep(0,special_iter)
# sdiff_ISTA=rep(0,special_iter)
#
# x0=rnorm(p)*rbinom(p,1,eps)
# y=A%*%x0+sigma*rnorm(n)
# lambda_seq=alpha_to_lambda(alpha,max_iter = 100,second_iter = 1000,sigma = sigma)
#
# # calculate FISTA iterates and minimizer
# SOLVER=SLOPE_solver_iter(A,y,lambda_seq,tol_infeas = 1e-15,tol_rel_gap = 1e-15,max_iter = 10000)
# slope=SOLVER$x[1:SOLVER$iter,]
# minimizer=slope[nrow(slope),]
# # support of true minimizer
# supp=which(minimizer!=0)
#
# # FISTA
# for (i in 1:special_iter){
#   sdiff_FISTA[i]=length(symdiff(which(abs(slope[i+1,])!=0),supp))
# }
#
# ### calculate AMP
# b=rep(0,p);z=y;tau=sqrt(sigma^2+eps/delta)
#
# for (i in 1:special_iter){
#   b=prox_sorted_L1(t(A)%*%z+b,alpha*tau)
#   z=y-A%*%b+z*length(unique(abs(b)))/n
#   tau=sqrt(F(tau,alpha,sigma = sigma,iteration = 1))
#   sdiff_AMP[i]=length(symdiff(which(b!=0),supp)) # should be \neq 0 but numerically you know
# }
#
#
# # calculate ISTA
# t=2/norm(A,'2')^2;b=rep(0,p)
# for (i in 1:special_iter){
#   b=prox_sorted_L1(b+t*t(A)%*%(y-A%*%b),t*lambda_seq)
#   sdiff_ISTA[i]=length(symdiff(which(abs(b)!=0),supp))
# }
#
# print('Hitting time at 0 set difference')
# print(min(which(sdiff_AMP==0)))
# print(min(which(sdiff_FISTA==0)))
# print(min(which(sdiff_ISTA==0)))
#
#
# # total time taken
# print(Sys.time()-super)
#
#
#
#
#
#
#
#
#
#
# ########## adjust for plot
# library(ggplot2)
# library(gridExtra)
# library(scales)
#
# record_AMP2=c(record_AMP,rep(tail(record_AMP,1),ISTA_iter-AMP_iter))
# record_FISTA2=c(record_FISTA,rep(tail(record_FISTA,1),ISTA_iter-FISTA_iter))
#
# df1=data.frame(iter =1:ISTA_iter, MSE = record_ISTA)
#
# df2=data.frame(iter =1:ISTA_iter, MSE = record_FISTA2)
#
# df3=data.frame(iter =1:ISTA_iter, MSE = record_AMP2)
#
# dat <- rbind(df3,df2,df1)
# dat$Algorithms <- rep(factor(c('AMP','FISTA','ISTA')),times=c(nrow(df1),nrow(df2),nrow(df3)))
#
# p1=ggplot(data = dat, aes(x = iter, y = MSE, colour = Algorithms,linetype=Algorithms)) +
#   geom_line(size=1)+ylab('Optimization Error')+
#   scale_linetype_manual(values=c("solid","dashed", "dotted"))+
#   scale_color_manual(values = c("red","blue", "black"))+
#   scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
#                 labels = trans_format("log10", math_format(10^.x)))+
#   theme(axis.title.x = element_blank(),
#         axis.title.y = element_text(size = rel(1.5)),
#         legend.title = element_blank(),
#         legend.text=element_text(size=15),
#         legend.key.size =unit(1,'cm'),
#         axis.text=element_text(size=15))
#
#
#
#
# df1=data.frame(iter =1:AMP_iter, set_diff = sdiff_ISTA)
# df2=data.frame(iter =1:AMP_iter, set_diff = sdiff_FISTA)
# df3=data.frame(iter =1:AMP_iter, set_diff = sdiff_AMP)
#
# dat <- rbind(df3,df2,df1)
# dat$Algorithms <- rep(factor(c('AMP','FISTA','ISTA')),times=c(nrow(df1),nrow(df2),nrow(df3)))
#
# p2=ggplot(data = dat, aes(x = iter, y = set_diff, colour = Algorithms)) +
#   geom_line(size=1,aes(linetype=Algorithms))+ylab('Set Difference')+
#   xlab('Iteration')+
#   scale_linetype_manual(values=c("solid","dashed", "dotted"))+
#   scale_color_manual(values = c("red","blue", "black"))+
#   scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
#                 labels = trans_format("log10", math_format(10^.x)))+
#   theme(axis.title.x = element_text(size = rel(1.6)),
#         axis.title.y = element_text(size = rel(1.5)),
#         legend.title = element_blank(),
#         legend.text=element_text(size=15),
#         legend.key.size =unit(1,'cm'),
#         axis.text=element_text(size=15))
#
# grid.arrange(p1,p2,ncol=1)
