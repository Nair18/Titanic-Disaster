#Kaggle Titanic Challenge

#This challenge is about applying machine learning techniques to
#predict a passenger's chance of surviving using R.

#reading the datasets

#Import the training set: train
train_url <- "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train <- read.csv(train_url) 
  
#Import the testing set: test
test_url <- "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test <- read.csv(test_url)
  
#Print train and test to the console
print(train)
print(test)

#attaching datasets
attach(train)
attach(test)

#loading some important libraries
library(dplyr)
library(stats)
library(utils)
library(ggplot2)
library(rpart)
library(randomForest)

#getting an overview of the training and testing data

#structure of train set
str(train)

#dimensions of train set
dim(train)

#first few entries of train set
head(train,4)

#descriptive statistics for train set
summary(train)

#structure of test set
str(test)

#dimensions of test set
dim(test)

#first few entries of test set
head(test,4)

#descriptive statistics for test set
summary(test)

#Begin asking meaningful queries from whatever you can see

#checking out people in our training set who survived the Titanic disaster

#passengers that survived vs passengers that passed away
base::table(train$Survived) 

#we see that 549 individuals died (62%) and 342 survived (38%)
#should we go for majority win prediction heuristic or dig deeper ?


#visualizing the same statistic as proportions
base::prop.table(base::table(train$Survived))


#table() command provides a convenient method to explore what variables 
#have predictive value. For example, maybe gender could play a role as 
#well here? let's explore this using the table() function for a 
#two-way comparison on the number of males and females that survived
base::table(train$Sex,train$Survived)

#as row-wise proportions,so specify 1 in the options
base::prop.table(base::table(train$Sex,train$Survived),1)
#as we look at the results, it appears as if it makes 
#sense to predict that all females will survive, and all men will die.

#Intelligent guessing also comes in handy (describe about domain knowledge)
#[example of property dealing, land house pricing]

#Does age play a role?
#Another variable that could influence survival is age: it's probable children 
#were saved first. You can test this by creating a new column with a categorical
#variable child. 

#child will take the value 1 in case age is <18, and a value of 0 in case age
#is >=18 and NA in case age is NA.

#To add this new variable you need to do two things 
#(i) create a new column, and (ii) provide the values for each observation 
#(i.e., row) based on the age of the passenger
train$Child[is.na(train$Age)] <- NA
train$Child[train$Age < 18] <- 1
train$Child[train$Age >=18] <- 0

#Now, we do a two-way comparison on the number of children vs adults
#that survived, in row-wise proportions to check if age does matter or not
base::prop.table(base::table(train$Child,train$Survived),1)

#Looking at our findings, we can say that while less obviously than gender, 
#age also seems to have an impact on survival.

#Making first predictions

#In our findings from the training set, females had over 50% chance of 
#surviving and males less than 50%. Hence, we can use this information
#for our first prediction: all females in the test set survive and 
#all males in the test set die.
#We use our test set for validating our predictions. 
#We might have seen that, contrary to the training set,
#the test set has no Survived column. 
#We add such a column using our predicted values. Next, when uploading our 
#results, Kaggle will use this variable (= our predictions) to score our 
#performance.

#Creating a copy of test: test_one
test_one <- test

#Initialize a Survived column to 0
test_one$Survived <- 0

#Set Survived to 1 if Sex equals "female"
test_one$Survived[test_one$Sex == "female"] <- 1

#So far, we did all the slicing and dicing of data to find subsets
#that have a higher chance of surviving

#Now let us get to some modeling
#We first go for decision tree

#A decision tree automates the kind of  process that we did so far, and 
#outputs a flowchart-like structure that is easy to interpret.

#Conceptually, the decision tree algorithm starts with all the data at the root node 
#and scans all the variables for the best one to split on. Once a variable is 
#chosen, you do the split and go down one level (or one node) and repeat. The 
#final nodes at the bottom of the decision tree are known as terminal nodes, 
#and the majority vote of the observations in that node determine 
#how to predict for new observations that end up in that terminal node.



#installing and loading up the rpart package for decision tree modelling
if(!require(rpart)){
  install.packages("rpart")
}
library(rpart)

#checking documentation for rpart to see how it works
?rpart
#method: Type of prediction you want. Here we predict a categorical 
#variable (dead or alive), so we're classifying: method = "class"

#Building tree model to predict survival based on the variables Passenger class,
#, Sex, Age, Number of Siblings/Spouses Aboard, 
#Number of Parents/Children Aboard, Passenger Fare and Port of Embarkation.
tree_model <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = train,method = "class" )

#Visuallize your tree using plot() and text()
plot(tree_model)
text(tree_model)

#if we see above graphs, they don't look highly informative and attracting
#So, we try and use some more useful packages which can offer us help 
#install & load the R packages rattle, rpart.plot, and RColorBrewer

#useful trick for multiple installation
pacman::p_load(rattle, rpart.plot, RColorBrewer)

#loading packages
library(rattle)
library(rpart.plot)
library(RColorBrewer)

#Use fancyRpartPlot() on tree_model to create a much fancier visualization 
#of tree
fancyRpartPlot(tree_model)

#Now, let us investigate our decision tree 
#find out the most important variables in decision tree
#(all shown in the tree except "Embarked" think why ?)
#(Best possible answer vs actually correct answer here)


#Since we have to predict results on test set:
#we pull out documentation of predict function
library(stats)
?predict
#predicting outcomes on test data 
predictions <- predict(tree_model,test,type = "class") #explain type option
#storing the solution in data frame as desired for submission
solution  <- data.frame(PassengerId = test$PassengerId,Survived = predictions)

#checking dimensions of solution to ensure all is fine
dim(solution)
#writing the solution to csv file
write.csv(solution, file="solution.csv",row.names = FALSE)
#viewing the decision tree solution
library(utils)
View(solution)

#Doing post-mortem analysis
#most common problem of decision trees -> overfitting
#(trying too hard to fit the data resulting in loss of generalization)
#in the above solution that we got by decision tree approach, 
#we obtained a result that outperforms a solution using purely gender

#Time for some theory:
#In rpart, the depth of our model is defined by two parameters:
#the cp parameter determines when the splitting up of the decision tree stops.
#the minsplit parameter monitors the amount of observations in a bucket. 
#If a certain threshold is not reached (e.g minimum 10 passengers) no further splitting can be done.

#trying out above two control options on our previously built tree model
#set the minsplit to 50 and the cp parameter to 0. 
#This can be done through the control argument of rpart() using rpart.control()
tree_model_1 <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = train,method = "class",control = rpart.control(cp = 0,minsplit = 50))
fancyRpartPlot(tree_model_1)

#Now we go for re-engineering our dataset
#(again domain knowledge & common sense comes in handy)
#A valid assumption is that larger families need more time to get 
#together on a sinking ship, and hence have less chance of surviving.
#Family size is determined by the variables SibSp and Parch, which 
#indicate the number of family members a certain passenger is traveling with. 
#So when doing feature engineering, you add a new variable family_size, which is 
#the sum of SibSp and Parch plus one (the observation itself), to the 
#test and train set

revised_train <- dplyr::mutate(train,family_sz = SibSp + Parch + 1)
revised_test <- dplyr::mutate(test,family_sz = SibSp + Parch + 1)

#Now creating yet another tree model from our revised training set after re-engineering of data
tree_model_2 <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = revised_train,method = "class",control = rpart.control(cp = 0,minsplit = 50))

#making our tree look informative, and interactive
fancyRpartPlot(tree_model_2)

#Again a learning curve for us:
#If you have a close look at your decision tree you see that family_size 
#is not included. Apparently other variables played a more important role. 
#This is part of the game as well. Sometimes things will not turn out as expected,
#but it is here that you can make the difference




#the Random Forest technique handles the overfitting problem you faced with decision trees.
#It grows multiple (very deep) classification trees using the training set. At the time of prediction, each tree is used to come up with a prediction and every outcome is counted as a vote. 
#For example, if you have trained 3 trees with 2 saying a passenger in the test set will survive and 1 says he will not, the passenger will be classified as a survivor. This approach of overtraining trees, but having the majority's vote count as the actual classification decision, avoids overfitting.
#Before starting with the actual analysis, you first need to meet one big condition of Random Forests: no missing values in your data frame. 

#majority voting replacement technique for categorical value
#median replacement technique for continuous value
# How to fill in missing Age values?
# We make a prediction of a passengers Age using the other variables and a decision tree model. 
# This time you give method = "anova" since you are predicting a continuous variable.

#One more important element in Random Forest is randomization to avoid the creation of the same tree from the training set. You randomize in two ways: by taking a randomized sample of the rows in your training set and by only working with a limited and changing number of the available variables for every node of the tree. Now it's time to do your first Random Forest analysis.