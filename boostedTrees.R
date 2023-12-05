library(tidyverse)
library(tidymodels, verbose = F)
library(modeltime)
library(timetk)
library(vroom)
library(embed)
library(bonsai)
library(lightgbm)
library(themis)


data = vroom("data.csv")
data2 = data %>% mutate(
  
  shot_made_flag = as.factor(shot_made_flag),
  #gamedate_team = str_c(game_date, opponent, sep = "_"),
  #season_team = str_c(season, opponent, sep = "_"),
  
) %>% select(c(shot_made_flag, shot_distance, shot_id, period, 
                seconds_remaining, shot_type))

# my_recipe <- recipe(shot_made_flag ~ ., data=train) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor)  %>% # turn all numeric features into factors5
#   # step_other(all_nominal_predictors(), threshold = .001) %>% 
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(shot_made_flag)) 
boosted_model <- boost_tree(tree_depth=1, #Determined by random store-item combos
                            trees=2000,
                            learn_rate=.1) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

train = data2 %>% filter(!is.na(shot_made_flag))
test = data2 %>% filter(is.na(shot_made_flag))

my_recipe <- recipe(shot_made_flag ~ ., data=train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor)  %>% # turn all numeric features into factors5
  step_other(all_nominal_predictors(), threshold = .01) %>% 
  #step_date(game_date) %>% 
  step_zv(all_predictors()) %>% 
  step_rm(shot_id) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(shot_made_flag)) %>% 
  step_corr(all_numeric_predictors(), threshold = .9) %>% 
  step_normalize(all_numeric_predictors()) #%>%   
  #step_smote(all_outcomes(), neighbors = 5)


# prepped = prep(my_recipe)
# x = bake(prepped, new_data = train)


boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boosted_model)

# 
# tuning_grid = grid_regular(tree_depth(),trees(), learn_rate())# idk what this does
# 
# folds = vfold_cv(train, v = 5, repeats = 1)
# 
# CV_results = boost_wf %>% tune_grid(resamples = folds, grid = tuning_grid,
#                                               metrics = metric_set(mn_log_loss))
# 
# bestTune = CV_results %>% select_best("mn_log_loss")
# 
final_wf = boost_wf %>% finalize_workflow(bestTune) %>% fit(train)


kobe_boost_preds = predict(final_wf, new_data = test, type = "prob")

sub = test %>% mutate(
  shot_made_flag = kobe_boost_preds$.pred_1,
  shot_id = shot_id
  
) %>% select(shot_id, shot_made_flag)


vroom_write(sub, "kobe_boost6.csv", delim = ",")



