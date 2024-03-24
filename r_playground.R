library(spam)
library(SuperLearner)
library(glmnet)
library(ranger)
library(xgboost)
library(e1071)
library(mvtnorm)
library(stats)
library(causalweight)

n=2000                            # sample size
p=100                             # number of covariates
s=2                               # number of covariates that are confounders
sigma=matrix(c(1,0.5,0.5,1),2,2)
e=(2*rmvnorm(n,rep(0,2),sigma))
x=matrix(rnorm(n*p),ncol=p)       # covariate matrix
beta=c(rep(0.25,s), rep(0,p-s))   # coefficients determining degree of confounding
d=(x%*%beta+rnorm(n)>0)*1         # treatment equation
z=rnorm(n)
s=(x%*%beta+0.25*d+z+e[,1]>0)*1   # selection equation
y=x%*%beta+0.5*d+e[,2]            # outcome equation
y[s==0]=0

# The true ATE is equal to 0.5
output=treatselDML(y,d,x,s)
cat("ATE: ",round(c(output$effect),3),", standard error: ",
     round(c(output$se),3), ", p-value: ",round(c(output$pval),3))
 output$ntrimmed

result <- c()
  
for (i in 1:100) {
  n=2000                            # sample size
  p=100                             # number of covariates
  s=2                               # number of covariates that are confounders
  sigma=matrix(c(1,0.5,0.5,1),2,2)
  e=(2*rmvnorm(n,rep(0,2),sigma))
  x=matrix(rnorm(n*p),ncol=p)       # covariate matrix
  beta=c(rep(0.25,s), rep(0,p-s))   # coefficients determining degree of confounding
  d=(x%*%beta+rnorm(n)>0)*1         # treatment equation
  z=rnorm(n)
  s=(x%*%beta+0.25*d+z+e[,1]>0)*1   # selection equation
  y=x%*%beta+0.5*d+e[,2]            # outcome equation
  y[s==0]=0
  
  output=treatselDML(y,d,x,s,z)
  result <- cbind(result, output$effect)
}

library(dplyr)
library(ggplot2)
matrix(result) %>% as.data.frame() %>% ggplot() + 
  geom_histogram(aes(x=V1), bins = 10, binwidth = 0.05)

summary(matrix(result))



