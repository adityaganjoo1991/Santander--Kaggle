#Normal Neural Network

library(e1071)
library(ROCR)
library(base)
library(psych)
library(Matrix)
library(caTools)
library(nnet) #for neural networks
library(caret) #for preprocessing

train<- read.csv("train.csv")
train.id = train$ID
train$ID=NULL #Removed ID from training dataset


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
TARGET= train3$TARGET
train3$TARGET=NULL

#normalizing before building model
procvalues= preProcess(train3,method=c("scale","center"))
train_normal2= predict(procvalues,train3)
train_normal2$TARGET=TARGET
set.seed(342)
model_nn= nnet(TARGET~.,data=train_normal2,size=3,rang=.1,decay=5e-4,maxit=300)
predicted= predict(model_nn,type='raw')
pred=prediction(predicted,train3$TARGET)
auc= as.numeric(performance(pred,"auc")@y.values)


Test= read.csv("test.csv")

#Removing the constant and redundant variables
Test2= subset(Test,select=names(Test) %ni% constant_train)
Test3= subset(Test2,select = names(Test2) %ni% toRemove) 
Test_ID= Test3$ID
Test3$ID=NULL


normalized_Test= predict(procvalues,Test3)
normalized_Test$TARGET=-1
pred_test= predict(model_nn,newdata=normalized_Test,type='raw')
submission= data.frame(ID=Test_ID,TARGET=pred_test)
write.csv(submission, "mysubmission23_NN_prop.csv",row.names=F)

#.5077371 with no scaling/centering. 
#.8190136 with scaling. On test in kaggle,  0.597769
# .74279 with normalization. On test in kaggle , .722838
# .8082 on train with normalization and svd reduction to 50 variables, On test in kaggle, .801747
# .8075785 on train with normalization and svd reduction to 150 variables
# .825671 on train with normalization and svd reduction to 120 variables, On test in kaggle, 0.805198
# .8156286 on train with normalization and svd reduction to 80 variables, On test in kaggle, 0.798856
# .8194631 on train with normalization and svd reduction to 100 variables
# .8246782 on train with normalization and svd reduction to 200 variables
# .8191215 on train with normalization and svd reduction to 180 variables
#.8153141 on train with normalization and svd reduction to 230 variables
#.8444972 on normalized train with comparable proportions of TARGET 1 and 0, on test in kaggle 0.813783
# .8487322 on normalized train with comparable TARGET proportions and SVD dimension reduction (230 variables), on test in kaggle 

















