library(rpart)
library(readxl)
library(randomForest)
library(pROC)
library(ROCR)
library(gbm)
gcDataF<- read_excel("D:/UIC/Fall 2017/Data Mining/Assignment/GermanCredit_assgt.xlsx")
head(gcDataF)
#colnames(gcDataF$`RADIO/TV`) <- "Radio_TV"
colnames(gcDataF)[colnames(gcDataF) == 'CO-APPLICANT'] <- 'CO_APPLICANT'
colnames(gcDataF)[colnames(gcDataF) == 'RADIO/TV'] <- 'Radio_TV'

catCols<-c("CHK_ACCT", "HISTORY", "NEW_CAR", "USED_CAR", "FURNITURE", "Radio_TV", "EDUCATION", "RETRAINING", "SAV_ACCT", "EMPLOYMENT", "MALE_DIV","MALE_SINGLE", "MALE_MAR_or_WID", "CO_APPLICANT", "GUARANTOR", "PRESENT_RESIDENT", "REAL_ESTATE", "PROP_UNKN_NONE", "OTHER_INSTALL", "RENT", "OWN_RES", "JOB", "TELEPHONE", "FOREIGN", "RESPONSE")
gcDataF[,catCols][is.na(gcDataF[,catCols])] <-0
gcDataF[,catCols]<-data.frame(apply(gcDataF[catCols], 2, as.factor))
head(gcDataF)
gcDataF$AGE[is.na(gcDataF$AGE)] <- round(mean(gcDataF$AGE, na.rm = TRUE))
nr=nrow(gcDataF)
trnIndex = sample(1:nr, size = round(0.7*nr), replace=FALSE) #get a random 70%sample of row-indices
gcDataF_trn=gcDataF[trnIndex,]   #training data with the randomly selected row-indices
gcDataF_tst = gcDataF[-trnIndex,]  #test data with the other row-indices

# random forest
set.seed(123)
rfModel_200 = randomForest(RESPONSE ~ ., data=gcDataF_trn, ntree=200, importance=TRUE )
rfModel_500 = randomForest(RESPONSE ~ ., data=gcDataF_trn, ntree=500, importance=TRUE )
rfModel_1000 = randomForest(RESPONSE ~ ., data=gcDataF_trn, ntree=1000, importance=TRUE )
#Variable importance  
importance(rfModel_200)
varImpPlot(rfModel_200)

importance(rfModel_500)
varImpPlot(rfModel_500)

importance(rfModel_1000)
varImpPlot(rfModel_1000)

#Validating Test
Pred_RF_200 <- predict(rfModel_200, gcDataF_tst, type='class')
Pred_RF_500 <- predict(rfModel_500, gcDataF_tst, type='class')
Pred_RF_1000 <- predict(rfModel_1000, gcDataF_tst, type='class')

#Confusion Matrix
table(pred= Pred_RF_200, true=gcDataF_tst$RESPONSE)
table(pred= Pred_RF_500, true=gcDataF_tst$RESPONSE)
table(pred= Pred_RF_1000, true=gcDataF_tst$RESPONSE)


#obtain the scores given yb the mode for the class of interest , here, the prob('good')
score_200=predict(rfModel_200,gcDataF_tst, type="prob")[,2] 
score_500=predict(rfModel_500,gcDataF_tst, type="prob")[,2] 
score_1000=predict(rfModel_1000,gcDataF_tst, type="prob")[,2] 
# the predict function  gives two scores, prob(0) and prob (1), and we keep the second column

#now apply the prediction function from ROCR to get a prrediction object
pred_200=prediction(score_200, gcDataF_tst$RESPONSE)
pred_500=prediction(score_500, gcDataF_tst$RESPONSE)
pred_1000=prediction(score_1000, gcDataF_tst$RESPONSE)
#obtain performance using the function from ROCR, then plot
perf_200=performance(pred_200, "tpr", "fpr")
perf_500=performance(pred_500, "tpr", "fpr")
perf_1000=performance(pred_1000, "tpr", "fpr")
plot(perf_200,col = 'red')
plot(perf_500, add = TRUE, col = 'blue')
plot(perf_1000, add = TRUE, col = 'green') 
legend(x="bottomright", y=NULL, legend=c("perf_200", "perf_500","perf_1000"),
       col=c("red", "blue","green"),lty=1:3,  cex=0.8)


#Obtain the AUC perfromance
auc_200=performance(pred_200, "auc")
auc_200
auc_500=performance(pred_500, "auc")
auc_500
auc_1000=performance(pred_1000, "auc")
auc_1000

#Obtain accuracy performance, and plot it vs different cutoff values
plot(performance(pred_200, "acc"),col = 'red')
plot(performance(pred_500, "acc"),add = TRUE,col = 'blue')
plot(performance(pred_1000, "acc"),add = TRUE,col = 'green')
legend(x="bottomright", y=NULL, legend=c("ACC_200", "ACC_500","ACC_1000"),
       col=c("red", "blue","green"),lty=1:3,  cex=0.8)


#Adaboost 
set.seed(123)

ada_model_200 <- gbm(RESPONSE ~ .,data=gcDataF_trn,distribution = "adaboost",n.trees = 200)
ada_model_500 <- gbm(RESPONSE ~ .,data=gcDataF_trn,distribution = "adaboost",n.trees = 500)
ada_model_1000 <- gbm(RESPONSE ~ .,data=gcDataF_trn,distribution = "adaboost",n.trees = 1000)

ada_200<-predict(ada_model_200,gcDataF_tst, type="response", n.trees = 200)
ada_500<-predict(ada_model_500,gcDataF_tst, type="response", n.trees = 500)
ada_1000<-predict(ada_model_1000,gcDataF_tst, type="response", n.trees = 1000)


#Draw the ROC curve for the Adaboost model
perf_ada_200= performance(prediction(ada_200, gcDataF_tst$RESPONSE), "tpr", "fpr")
perf_ada_500= performance(prediction(ada_500, gcDataF_tst$RESPONSE), "tpr", "fpr")
perf_ada_1000= performance(prediction(ada_1000, gcDataF_tst$RESPONSE), "tpr", "fpr")


plot(perf_ada_200,col="red")
plot(perf_ada_500,add=TRUE,col="blue")
plot(perf_ada_1000,add=TRUE,col="green")
legend(x="bottomright", y=NULL, legend=c("ROC_200", "ROC_500","ROC_1000"),
       col=c("red", "blue","green"),lty=1:3,  cex=0.8)



#Obtain the AUC perfromance
auc_ada_200<-performance(prediction(ada_200, gcDataF_tst$RESPONSE), "auc")
auc_ada_200
auc_ada_500<-performance(prediction(ada_500, gcDataF_tst$RESPONSE), "auc")
auc_ada_500
auc_ada_1000<-performance(prediction(ada_1000, gcDataF_tst$RESPONSE), "auc")
auc_ada_1000


#comparing ROC of random forest and Adaboost 
plot(perf_500,col="red")
plot(perf_ada_1000,add=TRUE,col="blue")
legend(x="bottomright", y=NULL, legend=c("Random Forest", "Adaboost"),
       col=c("red", "blue"),lty=1:2,  cex=0.8)


