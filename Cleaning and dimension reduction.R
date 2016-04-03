train<- read.csv("train.csv")
train.id = train$ID
train$ID=NULL #Removed ID from training dataset

#Dimension Reduction

#Identifying and removing features with constant values
names(train[, sapply(train, function(v) var(v, na.rm=TRUE)==0)]) #displays features with constant values
train2= train[,sapply(train, function(v) var(v, na.rm=TRUE)!=0)] #removes features with constant values
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

#Standardizing before SVD (Centering and Scaling)
train_standardized= scale(Train_Split,center=TRUE,scale=TRUE) #Scale true because features seem to be on different scales

#SVD dimension reduction
library(base)
train_svd<- svd(train_standardized) #229
plot(train_svd$d, ylab="Diagonal Values", main="Singular Value Decomposition")
plot(train_svd$d, ylim=c(0,1),ylab="Singular Values",main="Zoomed in representation of singular values") #Zoomed in plot to identify singular values above 1
#229 values above 1
d_filtered= train_svd$d[1:229] #selecting only the first 229 singular values which are above 1
D<- diag(d_filtered) #Creating its diagonal matrix for later calculation of reduced dataset
U<-train_svd$u[,1:229] #removed corresponding columns for u
V= train_svd$v[,1:229]
vt<- t(train_svd$v) #Transpose of v
VT<- vt[1:229,] #Removed corresponding rows of VT
train_reduced<- U %*% D #Reduced dimension dataset.
library(e1071)
train_rdf<- as.data.frame(train_reduced)
train_rdf$TARGET=Train_Split.target #230 features, including TARGET variable

#Reducing the split test data dimensions and adding TARGET for the purpose of predictions
Test_Split_Matrix= as.matrix(Test_Split)  #formatting as matrix for projecting points on principal directions
test_reduced= Test_Split_Matrix %*% V  #Projection of test data on principal axes
test_rdf= as.data.frame(test_reduced)  #converting back from matrix to dataframe
test_rdf$TARGET=Test_Split.target      #Adding the TARGET variable which had been removed for allowing projection on axes



#RandomForest on split train set with 230 features

# 1. ntree=10
set.seed(100)
forest_model_230 = randomForest(as.factor(TARGET)~.,data=train_rdf,ntree=10,proximity=FALSE)
pred_230= predict(forest_model_230,type="prob")
ROCRpred_230= prediction(pred_230[,2],train_rdf$TARGET)
auc_230= as.numeric(performance(ROCRpred_230,"auc")@y.values)
auc_230 #.583

# 2. ntree=50

set.seed(100)
forest_model_230_50 = randomForest(as.factor(TARGET)~.,data=train_rdf,ntree=50,proximity=FALSE)
pred_230_50= predict(forest_model_230_50,type="prob")
ROCRpred_230_50= prediction(pred_230_50[,2],train_rdf$TARGET)
auc_230_50= as.numeric(performance(ROCRpred_230_50,"auc")@y.values)
auc_230_50 #.6517
#using model with ntree= 50 on the split test data 
pred_test_230_50= predict(forest_model_230_50,newdata=test_rdf,type="prob")
ROCRpredtest_230_50=prediction(pred_test_230_50[,2],test_rdf$TARGET)
auc_test_230_50=as.numeric(performance(ROCRpredtest_230_50,"auc")@y.values)
auc_test_230_50 #.527664



# 3. ntree= 100
set.seed(100)
forest_model_230 = randomForest(as.factor(TARGET)~.,data=train_rdf,ntree=100,proximity=FALSE)
pred_230= predict(forest_model_230,type="prob")
ROCRpred_230= prediction(pred_230[,2],train_rdf$TARGET)
auc_230= as.numeric(performance(ROCRpred_230,"auc")@y.values)
auc_230 #.668 
#using model with ntree= 100 on the split test data 
pred_test_230= predict(forest_model_230,newdata=test_rdf,type="prob")
ROCRpredtest_230=prediction(pred_test_230[,2],test_rdf$TARGET)
auc_test_230=as.numeric(performance(ROCRpredtest_230,"auc")@y.values)
auc_test_230 #.5296

# 4. ntree= 120 # Didn't work. Could not allocate memory



#Randomforest on U with ntree=100 to check if there's a difference in auc value compared to UD
set.seed(100)
forest_model_U_100 = randomForest(as.factor(TARGET)~.,data=U_DF,ntree=100,proximity=FALSE)
pred_230_U_100= predict(forest_model_U_100,type="prob")
ROCRpred_230_U_100= prediction(pred_230_U_100[,2],train_rdf$TARGET)
auc_230_U_100= as.numeric(performance(ROCRpred_230_U_100,"auc")@y.values)
auc_230_U_100 #.6687. Comes out to be the same
#using model with ntree= 100 on the split test data 
pred_test_230_U_100= predict(forest_model_U_100,newdata=test_rdf,type="prob")
ROCRpredtest_230_U_100=prediction(pred_test_230_U_100[,2],test_rdf$TARGET)
auc_test_230_U_100=as.numeric(performance(ROCRpredtest_230_U_100,"auc")@y.values)
auc_test_230_U_100 #.5448065. A little higher than UD


#Scatterplot of U[,1] and U[,2] color coded by TARGET. This is to identify relationship between the eigenvectors and Target
library(ggplot2)
qplot(V1,V2,data=U_DF,color=factor(TARGET),xlab="First Vector of U",ylab="Second Vector of U", main= "Plot of U's Vectors, color coded by 'TARGET')
#Visualization isn't clear. 






**************************************************************************************

train_nt=train3 #train 3 contains the target variable
train_nt$TARGET=NULL 


#Rank of the matrix
library(Matrix)
Train_rank<- rankMatrix(train_nt) #209 independent variables

# centering the dataframe before SVD (subtracting column means)
train_nt_c= scale(train_nt,center=TRUE,scale=TRUE) #Scale true because features seem to be on different scales

# SVD dimension reduction
library(base)
train_svd<- svd(train_nt_c) #233
plot(train_svd$d, ylab="Diagonal Values", main="Singular Value Decomposition")
plot(train_svd$d, ylim=c(0,1),ylab="Singular Values",main="Zoomed in representation of singular values") #Zoomed in plot to identify singular values above 1
#233 values above 1
d_filtered= train_svd$d[1:233] #selecting only the first 233 singular values which are above 1
D<- diag(d_filtered) #Creating its diagonal matrix for later calculation of reduced dataset
U<-train_svd$u[,1:233] #removed corresponding columns for u
vt<- t(train_svd$v) #Transpose of v
VT<- vt[1:233,] #Removed corresponding rows of VT
train_reduced<- U %*% D #Reduced dimension dataset. Problem: Same no. of variables as before, i.e,308
library(e1071)
train_rdf<- as.data.frame(train_reduced)
train_rdf$TARGET=train3$TARGET


#Using 20 principal components
d_filtered= train_svd$d[1:20] #selecting only the first 231 singular values which are above 1
D<- diag(d_filtered) #Creating its diagonal matrix for later calculation of reduced dataset
U<-train_svd$u[,1:20] #removed corresponding columns for u
vt<- t(train_svd$v) #Transpose of v
VT<- vt[1:20,] #Removed corresponding rows of VT
train_reduced<- U %*% D #Reduced dimension dataset. Problem: Same no. of variables as before, i.e,308
library(e1071)
train_rdf<- as.data.frame(train_reduced)
train_rdf$TARGET=train3$TARGET

#randomForest on 20 feature whole train set
library(randomForest)
set.seed(100)
forest_model_20 = randomForest(as.factor(TARGET)~.,data=train_rdf,ntree=10,proximity=FALSE)
pred_20= predict(forest_model_20,type="prob")
library(ROCR)
ROCRpred_20= prediction(pred_20[,2],train_rdf$TARGET)
auc_20= as.numeric(performance(ROCRpred_20,"auc"))@y.values
auc #.58 (pretty bad)


