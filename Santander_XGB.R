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


#Splitting labelled data into train and test sets 
set.seed(100)
split=sample.split(train3$TARGET,SplitRatio=.75)
Train_Split= subset(train3,split==TRUE)
Test_Split= subset(train3,split==FALSE)
nrow(Train_Split)

#removing target variable before SVD
Train_Split.target= Train_Split$TARGET
Train_Split$TARGET=NULL
Test_Split.target=Test_Split$TARGET
Test_Split$TARGET=NULL

#Centering test and train before SVD (Centering and Scaling)
train_centered= scale(Train_Split,center=TRUE,scale=FALSE)
trainmean= colMeans(Train_Split)
test_centered= Test_Split
test_centered= as.matrix(test_centered)
Test_Split_matrix= as.matrix(Test_Split)

#For calculation of centered test
for(i in 1:ncol(Test_Split_matrix)) 
{
 for(j in 1:nrow(Test_Split_matrix)) 
 {
test_centered[j,i]= Test_Split_matrix[j,i] - trainmean[i]
 }
}
Test_Split_Matrix= test_centered

#SVD dimension reduction for Centered
train_svd<- svd(train_centered)

#function to calculate UDV by specifying number of features
Reduction = function(x) {
d_filtered= train_svd$d[1:x] #selecting only x singular values 
D<- diag(d_filtered) #Creating its diagonal matrix for later calculation of reduced dataset
U<-train_svd$u[,1:x] #removed corresponding columns for u
V= train_svd$v[,1:x]
vt<- t(train_svd$v) #Transpose of v
VT<- vt[1:x,] #Removed corresponding rows of VT
train_reduced<- U %*% D #Reduced dimension dataset.
train_rdf<- as.data.frame(train_reduced)
train_rdf$TARGET=Train_Split.target #x+1 features, including TARGET variable
#Reducing the split test data dimensions and adding TARGET for the purpose of predictions
test_reduced= Test_Split_Matrix %*% V  #Projection of test data on principal axes
test_rdf= as.data.frame(test_reduced)  #converting back from matrix to dataframe
test_rdf$TARGET=Test_Split.target      #Adding the TARGET variable which had been removed for allowing projection on axes

red = list("train"=train_rdf, "U"=U,"V"=V,"VT"=VT,"D"=D,"test"=test_rdf)  # List of objects to be returned
return(red)
}


red= Reduction(117)


red= Reduction(117)
train_sparse= sparse.model.matrix(TARGET~.,data=red$train)
dtrain <- xgb.DMatrix(data=train_sparse, label=red$train$TARGET)
watchlist<- list(train=dtrain)
param <- list(  objective           = "binary:logistic", 
                 booster             = "gbtree",
                 eval_metric         = "auc",
                 eta                 = 0.02,
                 max_depth           = 8,
                 subsample           = 0.9,
                 colsample_bytree    = 0.85
 )
> clf <- xgb.train(   params              = param, 
                      data                = dtrain, 
                      nrounds             = 350, 
                      verbose             = 1,
                      watchlist           = watchlist,
                      maximize            = FALSE
  )



Real_Test= read.csv("test.csv")
Test= Real_Test
#Removing the constant and redundant variables
Test2= subset(Test,select=names(Test) %ni% constant_train)
Test3= subset(Test2,select = names(Test2) %ni% toRemove) 
Test_ID= Test3$ID
Test3$ID=NULL
Test_matrix= as.matrix(Test3)
Real_Test_Centered= as.matrix(Test3)
 
#Centering real test data
for(i in 1:ncol(Test_matrix))
{
 for(j in 1:nrow(Test_matrix))
 { 
    Real_Test_Centered[j,i]= Test_matrix[j,i] - trainmean[i]
  }
 }
#Reducing Test dimensions
Test_reduced = Real_Test_Centered %*% red$V
Test_rdf= data.frame(Test_reduced)
Test_rdf$TARGET= -1
Test_Sparse= sparse.model.matrix(TARGET~.,data=Test_rdf)
pred_Test= predict(clf,Test_Sparse)
submission = data.frame(ID=Test_ID,TARGET=pred_Test)
write.csv(submission, "mysubmission2.csv",row.names=F)



#without splitting initial Train set
library(e1071)
library(ROCR)
library(randomForest)
library(base)
library(psych)
library(Matrix)
library(xgboost)

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

train4= train3


#removing target variable before SVD
train4.target= train4$TARGET
train4$TARGET=NULL


#Centering  train before SVD (Centering and Scaling)
train_centered= scale(train4,center=TRUE,scale=FALSE)
trainmean= colMeans(train4)

#SVD dimension reduction for Centered
train_svd<- svd(train_centered)

#function to calculate UDV by specifying number of features
Reduction = function(x) {
d_filtered= train_svd$d[1:x] #selecting only x singular values 
D<- diag(d_filtered) #Creating its diagonal matrix for later calculation of reduced dataset
U<-train_svd$u[,1:x] #removed corresponding columns for u
V= train_svd$v[,1:x]
vt<- t(train_svd$v) #Transpose of v
VT<- vt[1:x,] #Removed corresponding rows of VT
train_reduced<- U %*% D #Reduced dimension dataset.
train_rdf<- as.data.frame(train_reduced)
train_rdf$TARGET=train4.target #x+1 features, including TARGET variable
red = list("train"=train_rdf, "U"=U,"V"=V,"VT"=VT,"D"=D)  # List of objects to be returned
return(red)
}




red= Reduction(50)
train_sparse= sparse.model.matrix(TARGET~.,data=red$train)
dtrain <- xgb.DMatrix(data=train_sparse, label=red$train$TARGET)
watchlist<- list(train=dtrain)
param <- list(  objective           = "binary:logistic", 
                 booster             = "gbtree",
                 eval_metric         = "auc",
                 eta                 = 0.02,
                 max_depth           = 8,
                 subsample           = 0.9,
                 colsample_bytree    = 0.85
 )
clf <- xgb.train(   params              = param, 
                      data                = dtrain, 
                      nrounds             = 350, 
                      verbose             = 1,
                      watchlist           = watchlist,
                      maximize            = FALSE
  )


Real_Test= read.csv("test.csv")
Test= Real_Test
#Removing the constant and redundant variables
Test2= subset(Test,select=names(Test) %ni% constant_train)
Test3= subset(Test2,select = names(Test2) %ni% toRemove) 
Test_ID= Test3$ID
Test3$ID=NULL
Test_matrix= as.matrix(Test3)
Real_Test_Centered= as.matrix(Test3)
 
#Centering real test data
for(i in 1:ncol(Test_matrix))
{
 for(j in 1:nrow(Test_matrix))
 { 
    Real_Test_Centered[j,i]= Test_matrix[j,i] - trainmean[i]
  }
 }
#Reducing Test dimensions
Test_reduced = Real_Test_Centered %*% red$V
Test_rdf= data.frame(Test_reduced)
Test_rdf$TARGET= -1
Test_Sparse= sparse.model.matrix(TARGET~.,data=Test_rdf)
pred_Test= predict(clf,Test_Sparse)
submission = data.frame(ID=Test_ID,TARGET=pred_Test)
write.csv(submission, "mysubmission6_50.csv",row.names=F)


#with non reduced training set 

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

train_sparse= sparse.model.matrix(TARGET~.,data=train3)
dtrain <- xgb.DMatrix(data=train_sparse, label=train3$TARGET)
watchlist<- list(train=dtrain)
param <- list(  objective           = "binary:logistic", 
                 booster             = "gbtree",
                 eval_metric         = "auc",
                 eta                 = 0.02,
                 max_depth           = 8,
                 subsample           = 0.9,
                 colsample_bytree    = 0.85
 )
clf <- xgb.train(   params              = param, 
                      data                = dtrain, 
                      nrounds             = 350, 
                      verbose             = 1,
                      watchlist           = watchlist,
                      maximize            = FALSE
  )


Real_Test= read.csv("test.csv")
Test= Real_Test
#Removing the constant and redundant variables
Test2= subset(Test,select=names(Test) %ni% constant_train)
Test3= subset(Test2,select = names(Test2) %ni% toRemove) 
Test_ID= Test3$ID
Test3$ID=NULL
Test_matrix= as.matrix(Test3)
Real_Test_Centered= as.matrix(Test3)
Test3_Sparse= sparse.model.matrix(TARGET~.,data=Test3)
pred_Test3= predict(clf,Test3_Sparse)
submission = data.frame(ID=Test_ID,TARGET=pred_Test3)
write.csv(submission, "mysubmission7_307.csv",row.names=F)
 #.830148 with entire training set and 117 variables
 #.740993 with 50 variables on entire training set
 # .82 with split training set and 117 variables
# .838232 with all variables (exluding constant and identical) on entire training set.





