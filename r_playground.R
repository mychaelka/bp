library(spam)
library(SuperLearner)


# Creates either a machine learner or an ensemble method and returns the fitted model
# ybin = 1 means that it is a regression estimator
MLfunct=function(y, x, d1=NULL, d2=NULL, MLmethod="lasso",  ybin=0){
  if (is.null(d1)==0 & is.null(d2)==0) { y=y[d1==1 & d2==1]; x=x[d1==1 & d2==1,]}
  if (is.null(d1)==0 & is.null(d2)==1) { y=y[d1==1]; x=x[d1==1,]}
  if  (MLmethod=="lasso"){
    if (ybin==1) model=SuperLearner(Y = y, X = x, family = binomial(), SL.library = "SL.glmnet")
    if (ybin!=1) model=SuperLearner(Y = y, X = x, family = gaussian(), SL.library = "SL.glmnet")
  }
  if  (MLmethod=="randomforest"){
    if (ybin==1) model=SuperLearner(Y = y, X = x, family = binomial(), SL.library = "SL.ranger")
    if (ybin!=1) model=SuperLearner(Y = y, X = x, family = gaussian(), SL.library = "SL.ranger")
  }
  if  (MLmethod=="xgboost"){
    if (ybin==1) model=SuperLearner(Y = y, X = x, family = binomial(), SL.library = "SL.xgboost")
    if (ybin!=1) model=SuperLearner(Y = y, X = x, family = gaussian(), SL.library = "SL.xgboost")
  }
  if  (MLmethod=="svm"){
    if (ybin==1) model=SuperLearner(Y = y, X = x, family = binomial(), SL.library = "SL.svm")
    if (ybin!=1) model=SuperLearner(Y = y, X = x, family = gaussian(), SL.library = "SL.svm")
  }
  if  (MLmethod=="ensemble"){
    if (ybin==1) model=SuperLearner(Y = y, X = x, family = binomial(), SL.library = c("SL.glmnet", "SL.xgboost", "SL.svm", "SL.ranger"))
    if (ybin!=1) model=SuperLearner(Y = y, X = x, family = gaussian(), SL.library = c("SL.glmnet", "SL.xgboost", "SL.svm", "SL.ranger"))
  }
  if  (MLmethod=="parametric"){
    if (ybin==1) model=SuperLearner(Y = y, X = x, family = binomial(), SL.library = "SL.glm")
    if (ybin!=1) model=SuperLearner(Y = y, X = x, family = gaussian(), SL.library = "SL.lm")
  }
  model
} 


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

model <- MLfunct(y,x)
