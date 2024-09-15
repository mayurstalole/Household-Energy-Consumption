# installing necessory packages and libraries.
install.packages("relaimpo")
install.packages("polycor")
install.packages("kernlab")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("corrplot")
install.packages("C50")
install.packages("kernlab")
install.packages("caret")
library(dplyr)
library(caret)
library(tidyr)
library(randomForest)
library(ggplot2)
library(lime)
library(GGally)
library(performance)
library(MLmetrics)
library(lmtest)
library(car)
library(lubridate)
library(psych)
library(plotly)
library(corrplot)
library(forecast)
library(kernlab)
library(polycor)
library(C50)
library(gmodels)



# Set working directory
setwd(dirname(file.choose()))
getwd()

# Read data
df_energy <- read.csv("energydata.csv", stringsAsFactors = FALSE)

# Applying round func to only numeric columns
df_energy[sapply(df_energy, is.numeric)] <- lapply(df_energy[sapply(df_energy, is.numeric)], round, 2)

# You can view the structure or head of df_energy to confirm
str(df_energy)
head(df_energy)
summary(df_energy)


#Missing value
apply(df_energy, MARGIN = 2, FUN = function(x) sum(is.na(x)))
df_energy <- na.omit(df_energy)
df_energy

# Check the dimensions
dim(df_energy)

# Find duplicate rows
duplicates <- duplicated(df_energy)

# Count the number of duplicate rows
num_duplicates <- sum(duplicates)

# Print the number of duplicate rows
print(paste("There are", num_duplicates, "duplicate rows in the dataset."))


# boxplot of the variables
boxplot(df_energy[, 2], main = "Boxplot for dependent variable 1")

boxplot(df_energy[, 4:13], main = "Boxplot for Variables 4-10")
boxplot(df_energy[, 14:22], main = "Boxplot for Variables 11-19")
boxplot(df_energy[23])
boxplot(df_energy[, 24:28], main = "Boxplot for Variables 20-28")


#Correlation Analysis on independent variables

# Remove the target variable(s) - here we remove 'Appliances' and 'lights'
df_reduced <- df_energy[, !names(df_energy) %in% c("Appliances", "lights")]

# Select only numeric columns from the reduced dataset
numeric_data_reduced <- df_reduced[, sapply(df_reduced, is.numeric)]

# Calculate the correlation matrix for the reduced set of variables
cor_matrix_reduced <- cor(numeric_data_reduced, use = "complete.obs")
cor_matrix_reduced

# Visualize the correlation matrix without showing the correlation coefficients
corrplot(cor_matrix_reduced, method = "color", type = "upper", order = "hclust",
         tl.col = "black", tl.srt = 45,
         diag = FALSE)  

boxplot(df_energy$Appliances, xlab="appliances", ylab="Count", main = "Boxplot of appliances")
qqnorm(df_energy$Appliances, xlab = "appliances", ylab = "Count")
qqline(df_energy$Appliances, col=2) # red color
ks.test(df_energy$Appliances, pnorm, mean(df_energy$Appliances), sd(df_energy$Appliances))

################################################################################################

#Modelling

# Set the seed for reproducibility
set.seed(123)

# Calculate the number of rows to include in the training set
train_size <- floor(0.8 * nrow(df_energy))

# Create a random sample of row indices for the training set
train_indices <- sample(seq_len(nrow(df_energy)), size = train_size)


# Splitting of data
# Split the data into training and testing sets
energy.tr <- df_energy[train_indices, ]
energy.te <- df_energy[-train_indices, ]

# Display the sizes of the datasets to confirm the split
cat("Training set rows:", nrow(energy.tr), "\n")
cat("Testing set rows:", nrow(energy.te), "\n")



# min-max scaling
mm_energy <-  df_energy[-1]
# Apply min-max scaling to numeric columns
mm_energy[sapply(mm_energy, is.numeric)] <- lapply(mm_energy[sapply(mm_energy, is.numeric)], function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
})

boxplot(mm_energy)


# z-score

# Apply Z-score standardization to numeric columns
mm_energy[sapply(mm_energy, is.numeric)] <- lapply(mm_energy[sapply(mm_energy, is.numeric)], function(x) {
  (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
})

boxplot(mm_energy)


#############################################################################################
# Decision Tree

# Read data
df_energy <- read.csv("energydata.csv", stringsAsFactors = FALSE)

# Define cutoff points for "Appliances" energy consumption
low_usage <- 100
high_usage <- 250

# Create a new labeled factor variable for "Appliances"
df_energy$Appliances_label <- cut(df_energy$Appliances, 
                                  breaks = c(-Inf, low_usage, high_usage, Inf),
                                  labels = c("Low", "Medium", "High"),
                                  right = FALSE)

# Remove 4 unnecessary columns (Date, Appliances, Lights, RV2)
df_energy <- df_energy[c(-1, -2, -3, -29)]

# Min-Max scaling for numeric columns
num_cols <- sapply(df_energy, is.numeric)
df_energy[num_cols] <- lapply(df_energy[num_cols], function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
})
summary(df_energy[num_cols])

# Save the scaled data into a new CSV file
write.csv(df_energy, file = "new_energy_data.csv", row.names = FALSE)

# Read the scaled data
final_df_energy <- read.csv("new_energy_data.csv", stringsAsFactors = FALSE)

# Set the seed for reproducibility
set.seed(123)

# Calculate the number of rows to include in the training set
train_size_new <- floor(0.8 * nrow(final_df_energy))

# Create a random sample of row indices for the training set
train_indices_new <- sample(seq_len(nrow(final_df_energy)), size = train_size_new)

# Split the data into training and testing sets
new_energy.tr <- final_df_energy[train_indices_new, ]
new_energy.te <- final_df_energy[-train_indices_new, ]

# Display the sizes of the datasets to confirm the split
cat("Training set rows:", nrow(new_energy.tr), "\n")
cat("Testing set rows:", nrow(new_energy.te), "\n")

# check the proportion of class variable
prop.table(table(new_energy.tr$Appliances_label))
prop.table(table(new_energy.te$Appliances_label))

# Ensure that the 'Appliances_label' column is a factor with the desired levels
levels_desired <- c("High", "Low", "Medium")
new_energy.tr$Appliances_label <- factor(new_energy.tr$Appliances_label, levels = levels_desired)
new_energy.te$Appliances_label <- factor(new_energy.te$Appliances_label, levels = levels_desired)

# Modeling with C5.0 Decision Tree
model <- C5.0(x = new_energy.tr[, -which(names(new_energy.tr) == "Appliances_label")], 
              y = new_energy.tr$Appliances_label)


# Create a factor vector of predictions on test data
pred1 <- predict(model, new_energy.te, type = "class")
pred1 <- factor(pred1, levels = levels_desired)

# Cross tabulation of predicted versus actual classes
CrossTable(new_energy.te$Appliances_label, pred1,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

# Assume levels are "High", "Low", "Medium" as seen from your previous message
levels_desired <- c("High", "Low", "Medium")


# Convert the actual labels to factors with the same levels
new_energy.te$Appliances_label <- factor(new_energy.te$Appliances_label, levels = levels_desired)

# Convert the predicted labels and actual labels to factors with the same levels
pred1 <- factor(pred1, levels = levels_desired)
str(pred1)
table(pred1)

# Check the structure to confirm
str(pred1)
str(new_energy.te$Appliances_label)

# Compute the confusion matrix with 'High' as the positive class
conf_matrix <- confusionMatrix(pred1, new_energy.te$Appliances_label)
print(conf_matrix)


#############################################################################################


# boosting the accuracy of decision trees with 10 trials
set.seed(12345)
?C5.0Control
# Ensure that the 'Appliances_label' column is a factor

enerygy_boost10 <- C5.0(new_energy.tr[-26], 
                        new_energy.tr$Appliances_label, 
                        control = C5.0Control(minCases = 10), 
                        trials = 10)
enerygy_boost10
# summary(enerygy_boost10)

enerygy_boost_pred10 <- predict(enerygy_boost10, new_energy.te)
CrossTable(new_energy.te$Appliances_label, enerygy_boost_pred10,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

conf_matrix1 <- confusionMatrix(enerygy_boost_pred10, new_energy.te$Appliances_label, positive = "yes")
conf_matrix1

#############################################################################################

# Support Vector Machine.

?ksvm
# Apply min-max scaling
scale_min_max <- function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
}

new_energy.te[sapply(new_energy.te, is.numeric)] <- sapply(new_energy.te
                                                           [sapply(new_energy.te, is.numeric)], scale_min_max)

# Check if the scaling was applied correctly
summary(new_energy.te)


# run initial model with Vanilladot kernel, which is a Linear kernel.
set.seed(12345)
svm0 <- ksvm(Appliances_label ~ T1 + RH_1 + T2 + RH_2 + T3 + RH_3 + T4 + RH_4 + T5 + RH_5 + T6 + RH_6 + 
               T7 + RH_7 + T8 + RH_8 + T9 + RH_9 + T_out +	Press_mm_hg +	RH_out +	Windspeed	+ 
               Visibility +	Tdewpoint	+ rv1, data = new_energy.te, kernel = "vanilladot", type = "C-svc")


# look at basic information about the model
svm0

# evaluate
new_df_energy.pred0 <- predict(svm0, new_energy.te)
table(new_df_energy.pred0, new_energy.te$Appliances_label)
round(prop.table(table(new_df_energy.pred0, new_energy.te$Appliances_label))*100,1)

# sum diagonal for accuracy
sum(diag(round(prop.table(table(new_df_energy.pred0, new_energy.te$Appliances_label))*100,1)))


#Improvement of model by changing the kernel: radial basis -Gaussian
svm1 <- ksvm(Appliances_label ~ T1 + RH_1 + T2 + RH_2 + T3 + RH_3 + T4 + RH_4 + T5 + RH_5 + T6 + RH_6 + 
               T7 + RH_7 + T8 + RH_8 + T9 + RH_9 + T_out +	Press_mm_hg +	RH_out +	Windspeed	+ Visibility +	
               Tdewpoint	+ rv1, data = new_energy.te, kernel = "rbfdot", type = "C-svc")

# look at basic information about the model
svm1

# evaluate
new_df_energy.pred1 <- predict(svm1, new_energy.te)
table(new_df_energy.pred1, new_energy.te$Appliances_label)
round(prop.table(table(new_df_energy.pred1, new_energy.te$Appliances_label))*100,1)

# sum diagonal for accuracy
sum(diag(round(prop.table(table(new_df_energy.pred1, new_energy.te$Appliances_label))*100,1)))

new_df_energy.pred1 <- predict(svm1, new_energy.te)
conf_matrix2 <- confusionMatrix(new_df_energy.pred1, new_energy.te$Appliances_label)
print(conf_matrix2)
###################################################################################################

# Prepare and Print Performance Metrics
comparison <- data.frame(
  Model = c("Decision Tree", "SVM"),
  Accuracy = c(conf_matrix1$overall['Accuracy'], conf_matrix2$overall['Accuracy'])
)

print(comparison)

