library(tidyverse)
library(tidymodels, verbose = F)
library(modeltime)
library(timetk)
library(vroom)
library(embed)
library(bonsai)
library(lightgbm)
library(rpart)
library(themis)


data = vroom("data.csv")
data2 = data %>% mutate(
  
  shot_made_flag = as.factor(shot_made_flag),
  finalMinute = ifelse(minutes_remaining == 0, 1,0),
  trueBeater = ifelse(minutes_remaining == 0, ifelse(seconds_remaining <= 3,1,0),0),
  endOfQuarterHeave = if_else(period < 4 & shot_distance > 20 & seconds_remaining < 5,1,0)
  # period = as.factor(period),
  #season_team = str_c(season, opponent, sep = "_")
  
  
  
) %>% select(c(shot_made_flag, shot_distance, shot_id, period, 
               seconds_remaining, action_type, opponent, minutes_remaining,loc_x,loc_y,
               opponent,season, shot_zone_area, playoffs))

train = data2 %>% filter(!is.na(shot_made_flag))
test = data2 %>% filter(is.na(shot_made_flag))

my_recipe <- recipe(shot_made_flag ~ ., data=train) %>%
  #step_mutate_at(all_numeric_predictors(), fn = factor)  %>% # turn all numeric features into factors5
  #step_other(all_nominal_predictors(), threshold = .01) %>% 
  #step_date(game_date) %>% 
  step_zv(all_predictors()) %>% 
  step_rm(shot_id) %>% 
  #step_corr(all_numeric_predictors(), threshold = .7) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(shot_made_flag))  
#step_smote(all_outcomes(), neighbors = 5)



bartMod = bart(mode = "classification", engine = 'dbarts', trees = 22)

bartWf = workflow() %>% add_recipe(my_recipe) %>% add_model(bartMod) %>%
  fit(train)

bartPredict = bartWf %>% predict(new_data = test, type = "prob")

sub2 = test %>% mutate(
  shot_made_flag = bartPredict$.pred_1,
  shot_id = shot_id
  
) %>% select(shot_id, shot_made_flag)

bartStats = summary(sub2$shot_made_flag)
print(bartStats)

vroom_write(sub2, "kobe_bart1.csv", delim = ",")

