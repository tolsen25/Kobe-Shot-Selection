
library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(discrim)
library(yardstick)

data = vroom("data.csv")

loc_x_zero <- data$loc_x == 0
data['angle'] <- rep(0,nrow(data))
data$angle[!loc_x_zero] <- atan(data$loc_y[!loc_x_zero] / data$loc_x[!loc_x_zero])
data$angle[loc_x_zero] <- pi / 2

data2 = data %>% mutate(
  
  
  shot_made_flag = as.factor(shot_made_flag),
  matchup = ifelse(str_detect(matchup, 'vs.'), 'Home', 'Away'),
  time_remaining = minutes_remaining*60 + seconds_remaining,
  period = as.factor(period),
  shot_distance = sqrt((loc_x/10)^2 + (loc_y/10)^2), 
  season <- substr(str_split_fixed(season, '-',2)[,2],2,2)
  
  
) %>% select(c(shot_made_flag, shot_distance, shot_id, period, action_type, 
               opponent, time_remaining,season, playoffs, matchup,angle))


train = data2 %>% filter(!is.na(shot_made_flag))
test = data2 %>% filter(is.na(shot_made_flag))

# best Tune  == mtry =2 and min_n =40
treeModel = rand_forest(mtry = 2, min_n =  40, trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# my_recipe <- recipe(shot_made_flag ~ ., data=train) %>%
#   #step_mutate_at(all_numeric_predictors(), fn = factor)  %>% # turn all numeric features into factors5
#   #step_other(all_nominal_predictors(), threshold = .01) %>%
#   #step_date(game_date) %>%
#   step_zv(all_predictors()) %>%
#   step_rm(shot_id) %>%
#   #step_corr(all_numeric_predictors(), threshold = .7) %>%
#   step_normalize(all_numeric_predictors()) %>%
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(shot_made_flag))
# #step_smote(all_outcomes(), neighbors = 5)
# 

my_recipe <- recipe(shot_made_flag ~ ., data=train) %>%
  #step_zv(all_predictors()) %>%
  step_rm(shot_id) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_unknown(all_nominal_predictors()) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(shot_made_flag))
#step_dummy(all_nominal_predictors())



prepped = prep(my_recipe)
x = bake(prepped, new_data = train)

forestReg_workflow = workflow()  %>%
  add_recipe(my_recipe) %>% add_model(treeModel)

# tuning_grid = grid_regular(mtry(range = c(1,5)), min_n(), levels = 5)# idk what this does
# 
# folds = vfold_cv(train, v = 5, repeats = 1)
# 
# CV_results = forestReg_workflow %>% tune_grid(resamples = folds, grid = tuning_grid,
#                                               metrics = metric_set(mn_log_loss))
# 
# bestTune = CV_results %>% select_best("mn_log_loss")

final_wf = forestReg_workflow  %>% fit(train)


kobe_rf_preds2 = predict(final_wf, new_data = test, type = "prob")

sub = test %>% mutate(
  shot_made_flag = kobe_rf_preds2$.pred_1,
  shot_id = shot_id
  
) %>% select(shot_id, shot_made_flag)

kobeStats = summary(sub$shot_made_flag)
kobeStats

vroom_write(sub, "kobeRandomForest6.csv", delim = ",")

# testing -----------------------------------------------------------------

# library(randomForest)
# library(ranger)
# 
# # Assuming 'train_data' is your training dataset
# rf_model <- ranger(shot_made_flag ~ ., data = train, num.trees = 100, importance = "impurity")
# 
# # Extract feature importance
# feature_importance <- ranger::importance(rf_model)
# 
# # Print or visualize the feature importance
# print(feature_importance)
# 
# # Optionally, visualize the feature importance
# varImpPlot(rf_model)
# 
# 
# 
# data2 = data %>% mutate(
#   
#   shot_made_flag = as.factor(shot_made_flag),
#   
#   
# ) %>% select(c(shot_made_flag, shot_distance, shot_id, period, 
#                seconds_remaining, action_type, opponent, minutes_remaining,loc_x,loc_y,
#                opponent,season, shot_zone_area, playoffs,game_date))
# 
# 
# train = data2 %>% filter(!is.na(shot_made_flag))
# 











