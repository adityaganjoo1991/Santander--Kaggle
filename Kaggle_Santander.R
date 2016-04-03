library(e1071)
library(ROCR)
library(randomForest)
library(base)
library(psych)

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
library(caTools)
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


#Function to call random forest and return auc value for test and train

AUC= function(train,test,trees) {
set.seed(100)
forest_model = randomForest(as.factor(TARGET)~.,data=train,ntree=trees,proximity=FALSE)
pred= predict(forest_model,type="prob")
ROCRpred= prediction(pred[,2],train$TARGET)
auc_train= as.numeric(performance(ROCRpred,"auc")@y.values)
pred_test= predict(forest_model,newdata=test,type="prob")
ROCRpredtest=prediction(pred_test[,2],test$TARGET)
auc_test=as.numeric(performance(ROCRpredtest,"auc")@y.values)
auc= list("train"=auc_train,"test"=auc_test)
return(auc)
}



#without scaling random forest n =230 and trees=10, train auc =.6038 compared to .58 with scaling
#without scaling random forest n=230,trees=100, train auc=.699, test auc=.758
#without scaling random forest n=230, trees=100,train auc=.68, test auc=.74
#without scaling rf n=150, ntree=100, train auc= .707 , test auc=.7568
#without scaling rf n=180, ntree=100, train auc= .704 , test auc=.7556
#without scaling rf n=100, ntree=100, train auc= .7009 , test auc=.76
#without scaling rf n=80, ntree=100, train auc= .6867 , test auc=.748
#without scaling rf n=50, ntree=100, train auc= .6349 , test auc=.642
#without scaling rf n=110, ntree=100, train auc= .7127 , test auc=.7657
#without scaling rf n=130, ntree=100, train auc= .7138 , test auc=.7538
#without scaling rf n=125, ntree=100, train auc= .716 , test auc=.75509
#without scaling rf n=115, ntree=100, train auc= .7192 , test auc=.7708
#without scaling rf n=118, ntree=100, train auc= .7161 , test auc=.7649
#without scaling rf n=119, ntree=100, train auc= .7154 , test auc=.76357
#without scaling rf n=121, ntree=100, train auc= .7173544 , test auc=.763508
#without scaling rf n=125, ntree=100, train auc= .716 , test auc=.75509
#without scaling rf n=122, ntree=100, train auc= .7168 , test auc=.7689196
#without scaling rf n=116, ntree=100, train auc= .7165 , test auc=.7681
#without scaling rf n=120, ntree=100, train auc= .717 , test auc=.7711
#without scaling rf n=117, ntree=100, train auc= .7198 , test auc=.771

# Real Test
red= Reduction(117)
Rf_model= randomForest(as.factor(TARGET)~.,data=red$train,ntree=100,proximity=FALSE)
Test=read.csv("test.csv")
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


#Random Forest model with 100 trees and 117 features on Test
pred_Test= predict(Rf_model,newdata=Test_reduced,type="prob")
pred_Test= predict(Rf_model,newdata=Test_reduced,type="prob")
res= data.frame(ID=Test_ID,TARGET=pred_Test[,2])
write.csv(res,"submission1.csv",row.names=FALSE)


