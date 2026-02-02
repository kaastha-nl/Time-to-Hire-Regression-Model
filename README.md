## Project Overview
This project predicts time-to-hire (in days) for job vacancies using historical recruitment data.
The goal is to understand which factors influence hiring duration and evaluate how accurately it can be predicted using regression models.

## Objective
Can we estimate how long it will take to fill a vacancy based on early recruitment signals (applications, department, location, etc.)?

This insight helps HR teams:
plan recruitment timelines
identify bottlenecks
improve workforce planning

## Method
Feature engineering on recruitment lifecycle dates
Regression modeling to predict time_to_hire_days
Automated validation using test cases

## Model Performance
MAE: ~23 days
R²: ~0.54
This means the model predicts hiring duration within ~3 weeks on average.

## Tech Stack
Python (pandas, scikit-learn)

## Regression models
Feature engineering
PyTest for model validation
