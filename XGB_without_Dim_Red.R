library(e1071)
library(ROCR)
library(randomForest)
library(base)
library(psych)
library(Matrix)
library(xgboost)
library(caTools)


train<- read.csv("train.csv")
train.id = train$ID
train$ID=NULL #Removed ID from training dataset

#Dimension Reduction

#Identifying and removing features with constant values
names(train[, sapply(train, function(v) var(v, na.rm=TRUE)==0)]) #displays features with constant values
train2= train[,sapply(train, function(v) var(v, na.rm=TRUE)!=0)] #removes features with constant values
constant_train = names(train[, sapply(train, function(v) var(v, na.rm=TRUE)==0)])
# 34 features removed out of 370

#Removing identical features but keeping one copy in the dataset
features_pair <- combn(names(train2), 2, simplify = F) #creates all possible combinations of the features, 2 at a time, and generates a list
toRemove <- c()
for(pair in features_pair) {
f1 <- pair[1]
f2 <- pair[2]
if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
if (all(train2[[f1]]==train2[[f2]])) {
cat(f1, "and", f2, "are very highly correlated \n")
toRemove <- c(toRemove,f2) 
   }
  }
 }
`%ni%` <- Negate(`%in%`)
train3<- subset(train2,select = names(train2) %ni% toRemove) 
# 29 features removed out of 336

#Need to convert to sparse matrix for further creation into xgb.DMatrix
train_sparse= sparse.model.matrix(TARGET~.,data=train3)
#Need to convert to xgboost's own matrix, called DMatrix, for the purpose of using advanced parameters with xgb.train
dtrain <- xgb.DMatrix(data=train_sparse, label=train3$TARGET)
watchlist<- list(train=dtrain) #auc metric will be displayed for this when given as a parameter
param <- list(  objective           = "binary:logistic", 
                 booster             = "gbtree",
                 eval_metric         = "auc",
                 eta                 = 0.01,
                 max_depth           = 6,
                 subsample           = 0.7,
                 colsample_bytree    = 0.7
 )


clf <- xgb.train(   params              = param, 
                      data                = dtrain, 
                      nrounds             = 599, 
                      verbose             = 1,
                      watchlist           = watchlist,
                      maximize            = FALSE
  )


#cross-validate xgboost to get the optimum number of nrounds
clf_cv = xgb.cv(  params = param,
                  data = dtrain,
                  nrounds = 600, 
                  nfold = 5,                                                   # number of folds in K-fold
                  prediction = TRUE,                                           # return the prediction using the final model 
                  watchlist = watchlist,
                  maximize = FALSE,                                                            #le is unbalanced; use stratified sampling
                  verbose = 1,
                 
)
#clf_cv returns a list containing a dataframe comprising of test auc mean and train auc mean values, and also the prediction values. 
#clf_cv$dt represents the dataframe mentioned above
dt=clf_cv$dt 
max(dt$test.auc.mean)#what's the max value of the test mean auc.
which.max(dt$test.auc.mean) # for which nround number did we get this max value
#now we have the optimum nrounds, we'll create the model on train data using xgb.train and use that model to make predictions on the real test data

Test= read.csv("test.csv")

#Removing the constant and redundant variables
Test2= subset(Test,select=names(Test) %ni% constant_train)
Test3= subset(Test2,select = names(Test2) %ni% toRemove) 
Test_ID= Test3$ID
Test3$ID=NULL
Test3$TARGET= -1
Test3_Sparse= sparse.model.matrix(TARGET~.,data=Test3)
pred_Test3= predict(clf,Test3_Sparse)
submission = data.frame(ID=Test_ID,TARGET=pred_Test3)
write.csv(submission, "mysubmission15_cv_307.csv",row.names=F)

#0.830148 with entire training set and 117 variables
#0.740993 with 50 variables on entire training set
#0.82 with split training set and 117 variables
#0.838232 with all variables (excluding constant and identical) on entire training set. nrounds = 350
#0.838216 with  all variables (excluding constant and identical) on entire training set. nrounds = 500
#0.836451 with  all variables (excluding constant and identical) on entire training set. nrounds = 350, eta=.01
#0.838421 with  all variables (excluding constant and identical) on entire training set. nrounds = 350, subsample=.8	
#0.838540 with  all variables (excluding constant and identical) on entire training set. nrounds = 350, subsample=.7	
#0.838785 with cross validation (nround 275) and nfolds = 5,nrounds = 275,subsample=.7
#0.839747 with cross validation (nround 599) and nfolds=5, eval_metric= "auc",eta= 0.01,max_depth= 6,subsample= 0.7,colsample_bytree= 0.7 (mysubmission15_cv_307)
