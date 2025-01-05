# 1. Introduction ----
# ____________________

# Read of PhysicalSound, a UK-based e-commerce company
#setwd("C:/Users/luish/OneDrive - University of Leeds/3. LUBS5308M01 Bus Analyt Decision Sci/coursework1")
data <- read.csv("order_july24.csv", header = TRUE, stringsAsFactors = TRUE)
# Libraries
library(mice)
library(caret)

# 2. Data Understanding ----
# ___________________________

summary(data)
data$voucher <- as.factor(data$voucher)
data$ad_channel <- as.factor(data$ad_channel)
summary(data)

# Select numeric and categorical variables
numeric_data <- data[sapply(data, is.numeric)]
categorical_data <- data[sapply(data, is.factor)]

# Plot boxplots for numeric variables
par(mfrow = c(1, 4))
par(mar = c(3, 3, 3, 3)) 
for (i in 1:ncol(numeric_data)) {
  boxplot(numeric_data[[i]], main = colnames(numeric_data)[i], 
          ylab = colnames(numeric_data)[i], col = "lightblue")
}

# Plot bar plots for categorical variables
par(mfrow = c(1, 2))
for (i in 1:ncol(categorical_data)) {
  barplot(table(categorical_data[[i]]), main = colnames(categorical_data)[i], 
          ylab = "Frequency", col = "#F4A582")
}

# Reset the plotting area to a single panel
par(mfrow = c(1, 1))
reset.par()

# 3. Exploratory Data Analysis (EDA) ----
# ________________________________________

# Histogram of spend
hist(data$spend)

# Set up the plotting area for 2 plots in one row
par(mfrow = c(1, 2))

# Scatter plot: Spend vs Time on Web with trendline
plot(data$time_web, data$spend,
     xlab = "Time on Web (time_web)", 
     ylab = "Spend",
     main = "Spend vs Time on Web",
     col = "blue", pch = 16)
# Add a linear regression trendline
abline(lm(spend ~ time_web, data = data), col = "red", lwd = 2)

# Scatter plot: Spend vs Age with trendline
plot(data$age, data$spend,
     xlab = "Age", 
     ylab = "Spend",
     main = "Spend vs Age",
     col = "blue", pch = 16)
# Add a linear regression trendline
abline(lm(spend ~ age, data = data), col = "red", lwd = 2)

# Reset the plotting area to a single panel
par(mfrow = c(1, 1))

# 4. Data Preparation ----
# _________________________

# Missing data analysis
library("VIM")
par(mar = c(10, 10, 10, 10)) 
aggr(data, numbers = TRUE, prop = FALSE, las = 2, cex.axis = 0.8)
par(mar = c(4, 4, 4, 4)) 

# Correlation analysis 
data$voucher <- as.numeric(data$voucher)
library("corrgram")
corrgram(data, order = NULL)

# Correlation with missing data
data_with_miss <- data
data_with_miss$missing <- as.numeric(!complete.cases(data))
corrgram(data_with_miss)


# Split Data into Training and Testing
set.seed(123)  # For reproducibility
train_indices <- sample(1:nrow(data), size = 0.8 * nrow(data))  # 80% training data
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# 5. Predictive Modelling ----
# __________________________

# Feature Selection: Adding or Removing Variables
# List of variable combinations to evaluate
variable_sets <- list(
  #base_model = c('past_spend', 'age', 'time_web')
  #with_ad_channel = c('past_spend', 'age', 'time_web', 'ad_channel')
  #with_voucher = c('past_spend', 'age', 'time_web', 'voucher')
  full_model = c('past_spend', 'age', 'time_web', 'ad_channel','voucher')
)

# Initialize results dataframe
feature_results <- data.frame(Model = character(), MAE = numeric(), MSE = numeric(), RMSE = numeric(), R_squared = numeric())

# selecting variables for the model
model_name = "full_model"
variables <- variable_sets[[model_name]]

# Multiple imputation on training data
mi_object <- mice(subset(train_data, select = c('spend', variables)), 
                  m = 10, maxit = 10, seed = 123)

# Fit the model
fit <- with(data = mi_object, exp = lm(as.formula(paste('spend ~', paste(variables, collapse = '+')))))
pooled_model <- pool(fit)
pooled_coeffs1 <- summary(pooled_model)$estimate

# Impute Missing Values in Test Data
test_imputed <- complete(mice(subset(test_data,
                                     select = c('spend', 'past_spend', 'age', 'time_web', 'ad_channel', 'voucher')), 
                              m = 1, maxit = 5))

# Predictions
test_matrix <- model.matrix(as.formula(paste('~', paste(variables, collapse = '+'))), data = test_imputed)
names(pooled_coeffs1) <- c("(Intercept)", "past_spend", "age", "time_web","ad_channel2", "ad_channel3", "ad_channel4", "voucher")
predictions <- test_matrix[, names(pooled_coeffs1)] %*% pooled_coeffs1

# Calculate performance metrics
mae <- mean(abs(test_imputed$spend - predictions))
mse <- mean((test_imputed$spend - predictions)^2)
rmse <- sqrt(mse)
r_squared <- 1 - sum((test_imputed$spend - predictions)^2) / sum((test_imputed$spend - mean(test_imputed$spend))^2)

# Save results
feature_results <- rbind(feature_results, data.frame(Model = model_name, MAE = mae, MSE = mse, RMSE = rmse, R_squared = r_squared))

# Summary of feature selection results
summary(pooled_model)
cat("Feature Selection Results:\n")
print(feature_results)

# 6. Evaluation: Cross Validation for the model's performance ----
# ________________________________________________________________

# Define the number of splits
n_splits <- 5
set.seed(123)

# Create indices for train-test splits
#split_indices <- createDataPartition(data$spend[!is.na(data$spend)], p = 0.8, list = FALSE, times = n_splits)
split_indices <- list()
for (i in 1:n_splits) {
  split_indices[[i]] <- sample(1:nrow(data), size = 0.8 * nrow(data))
}

# Initialize results dataframe
cv_results <- data.frame(Split = integer(), MAE = numeric(), MSE = numeric(), RMSE = numeric(), R_squared = numeric())

# Loop over train-test splits
for (i in 1:n_splits) {
  # Train-test split
  train_data <- data[split_indices[[i]], ]
  test_data <- data[-split_indices[[i]], ]
  
  # Multiple imputation on training data
  mi_object <- mice(subset(train_data,
                           select = c('spend', 'past_spend', 'age', 'time_web', 'ad_channel', 'voucher')), 
                    m = 10, maxit = 10, seed = 123)
  
  # Fit the model
  fit <- with(data = mi_object, exp = lm(spend ~ past_spend + age + time_web + ad_channel + voucher))
  pooled_model <- pool(fit)
  pooled_coeffs <- summary(pooled_model)$estimate
  
  # Impute test data
  test_imputed <- complete(mice(subset(test_data,
                                       select = c('spend', 'past_spend', 'age', 'time_web', 'ad_channel', 'voucher')),
                                m = 1, maxit = 5, seed = 123))
  # Predictions
  test_matrix <- model.matrix(~ past_spend + age + time_web + ad_channel + voucher, data = test_imputed)
  names(pooled_coeffs) <- c("(Intercept)", "past_spend", "age", "time_web", 
                            "ad_channel2", "ad_channel3", "ad_channel4", "voucher")
  predictions <- test_matrix[, names(pooled_coeffs)] %*% pooled_coeffs
  
  # Calculate performance metrics
  mae <- mean(abs(test_imputed$spend - predictions))
  mse <- mean((test_imputed$spend - predictions)^2)
  rmse <- sqrt(mse)
  r_squared <- 1 - sum((test_imputed$spend - predictions)^2) / sum((test_imputed$spend - mean(test_imputed$spend))^2)
  
  # Save results
  cv_results <- rbind(cv_results, data.frame(Split = i, MAE = mae, MSE = mse, RMSE = rmse, R_squared = r_squared))
}

# Summary of cross-validation results
cat("Cross-Validation Results:\n")
print(cv_results)

# Average performance metrics across splits
avg_results <- colMeans(cv_results[, -1])
cat("Average Performance Across Splits:\n")
print(avg_results)

# 7. Predicting the spending of 20 new customers ----
# ____________________________________________________

# Read test data
test_data <- read.csv("new_customer24.csv", header = TRUE, stringsAsFactors = TRUE)

# Ensure test data matches training data levels for categorical variables
#test_data$voucher <- as.factor(test_data$voucher)  # Convert to factor
test_data$ad_channel <- as.factor(test_data$ad_channel)  # Convert to factor

# Align levels with training data
#test_data$voucher <- factor(test_data$voucher, levels = levels(data$voucher))
test_data$ad_channel <- factor(test_data$ad_channel, levels = levels(data$ad_channel))

# Create model matrix for test data
test_matrix <- model.matrix(~ past_spend + age + time_web + ad_channel + voucher, 
                            data = test_data)

# Extract pooled coefficients for prediction (including intercept)
pooled_coeffs <- pooled_coeffs1

# Ensure dimensions match (names of coefficients align with columns of test_matrix)
if (!all(names(pooled_coeffs) %in% colnames(test_matrix))) {
  stop("Mismatch between model coefficients and test data variables.")
}

# Make predictions on the test data
names(pooled_coeffs) <- c("(Intercept)", "past_spend", "age", "time_web", 
                          "ad_channel2", "ad_channel3", "ad_channel4", "voucher")
predictions <- test_matrix[, names(pooled_coeffs)] %*% pooled_coeffs

# Output predictions
print(predictions)

