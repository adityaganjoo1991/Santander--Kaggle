library(e1071)
library(ROCR)
library(base)
library(psych)
library(Matrix)
library(caTools)
library(nnet) #for neural networks
library(caret) #for preprocessing


#comparable number of 1's and 0's in target

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

#subsetting target=1
train_T1 = subset(train3, TARGET==1)
#3000 rows

#subsetting around 7000 rows of target=0
train_T0= subset(train3,TARGET==0)

train_T00= train_T0[1:7000,]

#combining
train_prop= rbind(train_T1,train_T00)

#shuffling the values
train_ps=  train_prop[sample(nrow(train_prop)),]

TARGET= train_ps$TARGET
train_ps$TARGET=NULL

#normalizing before building model
procvalues= preProcess(train_ps,method=c("scale","center"))
train_normal= predict(procvalues,train_ps)
train_normal$TARGET=TARGET
train_normal$TARGET=NULL

#SVD dimension reduction
train_svd= svd(train_normal)



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
train_rdf$TARGET=TARGET #x+1 features, including TARGET variable
red = list("train"=train_rdf, "U"=U,"V"=V,"VT"=VT,"D"=D)  # List of objects to be returned
return(red)
}

red=Reduction(230)

numFolds = trainControl(method="cv",number=10, summaryFunction=twoClassSummary,classProbs=TRUE)
nngrid= expand.grid(.size=c(1,3,5), .decay=c(0,0.001,0.1,5e-4)) #takes all possible combinations of specified values

model_cv= train(TARGET~.,data=red$train, method= 'nnet',maxit=300,trControl=numFolds,tuneGrid=nngrid)
#Models compared on auc values. According to cv, best result with size=3 and decay =.1

model_nn= nnet(TARGET~.,data=red$train,size=3,rang=.1,decay=0.1,maxit=300)
predicted= predict(model_nn,type='raw')
pred=prediction(predicted,red$train$TARGET)
auc= as.numeric(performance(pred,"auc")@y.values)


Test= read.csv("test.csv")

#Removing the constant and redundant variables
Test2= subset(Test,select=names(Test) %ni% constant_train)
Test3= subset(Test2,select = names(Test2) %ni% toRemove) 
Test_ID= Test3$ID
Test3$ID=NULL
normalized_Test= predict(procvalues,Test3)
pred_test= predict(model_nn,newdata=normalized_Test,type='raw')
submission= data.frame(ID=Test_ID,TARGET=pred_test)
write.csv(submission, "mysubmission29_NN_prop_230_CV.csv",row.names=F)
#$ 0.851177 on normalized train with comparable TARGET proportions and SVD dimension reduction(230 variables), on test in kaggle 0.818680

