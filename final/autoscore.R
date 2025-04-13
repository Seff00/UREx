# Load necessary libraries
library(AutoScore)
library(dplyr)

# Load training data
data_train <- read.csv("final/data_train_infection.csv")
names(data_train)[names(data_train) == "infection"] <- "label"
data_train$label <- ifelse(data_train$label == 1, 1, 0)

# Variable ranking
ranked_var <- AutoScore_rank(train_set = data_train)

final_vars <- names(ranked_var)[1:20]

# Generate score tables
cut_vec_object <- AutoScore_weighting(
  train_set = data_train,
  validation_set = data_train,
  final_variables = final_vars
)

# --- Manually Generate the Score Table using cut_vec ---

# Step 1: Subset the training data to final variables + label
train_set_1 <- data_train[, c(final_vars, "label")]

# Step 2 & 3:
# Transform data and compute scores using internal AutoScore functions
score_vector <- tryCatch({
  # Step 2: Transform data using the captured cut_vec
  print("Applying variable transformations using cut_vec...")
  # Ensure AutoScore is loaded for ::: access
  if (!"AutoScore" %in% .packages()) library(AutoScore)
  train_set_2 <- AutoScore:::transform_df_fixed(
    train_set_1, cut_vec = cut_vec_object
  )

  # Step 3: Compute the score table (returns a named vector of scores)
  print("Computing score points using transformed data...")
  score_tbl_vector <- AutoScore:::compute_score_table(
    train_set_2 = train_set_2,
    max_score = 100,
    variable_list = final_vars
  )
  score_tbl_vector # Return the named vector
}, error = function(e) {
  print("--- ERROR during manual transformation or score computation ---")
  print(e)
  NULL # Return NULL on error
})

# --- Export the Computed Score Table ---

# Check if score_vector was created successfully
print(head(score_vector))

# Step 4: Convert the named vector score_vector to a two-column data frame
score_df_to_export <- data.frame(
  Variable_Category = names(score_vector),
  Score = score_vector,
  row.names = NULL,      # Prevent writing R row names
  check.names = FALSE # Prevent converting special characters in variable names
)

# Step 5: Write the data frame to CSV
output_filename <- "final/autoscore_output.csv"
write.csv(score_df_to_export, output_filename, row.names = FALSE, quote = TRUE)
