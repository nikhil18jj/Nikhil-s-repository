library(lars) 
library(glmnet) 
library(caret)
data(diabetes) 
attach(diabetes)
set.seed(1234)
par(mfrow=c(2,5))
for(i in 1:10){   plot(x[,i], y) 
abline(lm(y~x[,i])) } 
layout(1) 

model_ols <- lm(y ~ x) 
summary(model_ols) 
#ridge regression #2 
lambdas <- 10^seq(7, -3) 
model_ridge <- glmnet(x, y, alpha = 0, lambda = lambdas) 
plot(model_ridge, xvar = "lambda", label = TRUE) 
#3 
cv_fit <- cv.glmnet(x=x, y=y, alpha = 0, nlambda = 1000)
plot(cv_fit) 
cv_fit$lambda.min 
#4 
fit <- glmnet(x=x, y=y, alpha = 0, lambda=cv_fit$lambda.min) 
fit$beta 
#5
fit <- glmnet(x=x, y=y, alpha = 0, lambda=cv_fit$lambda.1se) 
fit$beta 
intrain <- createDataPartition(y=diabetes$y,p = 0.8,list = FALSE) 
training <- diabetes[intrain,] 
testing <- diabetes[-intrain,] 
cv_ridge <- cv.glmnet(x=training$x, y=training$y,alpha = 0, nlambda = 1000) 
ridge_reg <- glmnet(x=training$x, y=training$y,alpha = 0, lambda=cv_ridge$lambda.min) 
ridge_reg$beta 
ridge_reg <- glmnet(x=training$x, y=training$y,alpha = 0, lambda=cv_ridge$lambda.1se) 
ridge_reg$beta 
#8 
ridge_reg <- glmnet(x=training$x, y=training$y,alpha = 0, lambda=cv_ridge$lambda.min)
ridge_pred <- predict.glmnet(ridge_reg,s = cv_ridge$lambda.min, newx = testing$x) 
sd((ridge_pred - testing$y)^2)/sqrt(length(testing$y)) 
ridge_reg <- glmnet(x=training$x, y=training$y,alpha = 0, lambda=cv_ridge$lambda.1se) 
ridge_pred <- predict.glmnet(ridge_reg,s = cv_ridge$lambda.1se, newx = testing$x) 
sd((ridge_pred - testing$y)^2)/sqrt(length(testing$y)) 
#9 
ols_reg <- lm(y ~ x, data = training) 
summary(ols_reg) 
#10 
ols_pred <- predict(ols_reg, newdata=testing$x, type = "response") 
sd((ols_pred - testing$y)^2)/sqrt(length(testing$y))
#least squares prediction error is higher. 

#lasso regression 
#2 
llambdas <- 10^seq(7, -3) 
model_ridgel <- glmnet(x, y, alpha = 1, lambda = llambdas) 
plot(model_ridgel, xvar = "lambda", label = TRUE) 
#3 
cv_fitl <- cv.glmnet(x=x, y=y, alpha = 1, nlambda = 1000) 
plot(cv_fitl) 
#4 
cv_fitl$lambda.min 
fitl <- glmnet(x=x, y=y, alpha = 1, lambda=cv_fitl$lambda.min)
fitl$beta 
#5 
fitl <- glmnet(x=x, y=y, alpha = 1, lambda=cv_fitl$lambda.1se) 
fitl$beta 
#6 
intrainl <- createDataPartition(y=diabetes$y, p = 0.8,list = FALSE)
trainingl <- diabetes[intrainl,] 
testingl <- diabetes[-intrainl,] 
#7 
cv_lasso <- cv.glmnet(x=trainingl$x, y=trainingl$y,alpha = 1, nlambda = 1000) 
lasso_reg <- glmnet(x=trainingl$x, y=trainingl$y,alpha = 1, lambda=cv_lasso$lambda.min) 
lasso_reg$beta 
lasso_reg <- glmnet(x=trainingl$x, y=trainingl$y,alpha = 1, lambda=cv_lasso$lambda.1se) 
lasso_reg$beta 
#8 
lasso_reg <- glmnet(x=trainingl$x, y=trainingl$y,alpha = 1, lambda=cv_lasso$lambda.min) 
lasso_pred <- predict.glmnet(lasso_reg,s = cv_lasso$lambda.min, newx = testingl$x) 
sd((lasso_pred- testingl$y)^2)/sqrt(length(testingl$y)) 
lasso_reg <- glmnet(x=trainingl$x, y=trainingl$y, alpha = 1, lambda=cv_lasso$lambda.1se) 
lasso_pred <- predict.glmnet(lasso_reg,s = cv_lasso$lambda.1se, newx = testingl$x) 
sd((lasso_pred - testingl$y)^2)/sqrt(length(testingl$y)) 
