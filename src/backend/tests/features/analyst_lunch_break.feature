Feature: Analyst Lunch Break Flow
  A business analyst with no coding experience uploads their quarterly sales data
  and, in a single lunch break, arrives at a deployed prediction model with a
  live dashboard they can share with their VP. Every step should work through
  natural conversation — no forms, no configuration, no code.

  Background:
    Given a new project named "Q4 Revenue Analysis"
    And a quarterly sales CSV is uploaded to the project

  Scenario: Upload reveals immediate data insight
    Then the dataset has 12 rows and 4 columns
    And the column names include "region", "product", "units", and "revenue"
    And each numeric column has min, max, and mean statistics
    And the dataset profile is cached for instant retrieval

  Scenario: Analyst explores data by asking questions
    When the analyst asks "what are the top performing regions?"
    Then the chat response contains a natural language answer
    And the response does not contain a stack trace

  Scenario: Training a model with a named target succeeds
    Given a feature set targeting "revenue" exists for the dataset
    When the analyst trains a Linear Regression model to predict "revenue"
    Then a model run is created with status "done"
    And the run has an R² metric above zero
    And the run records train and test set sizes

  Scenario: Deploying the trained model produces a working endpoint
    Given a feature set targeting "revenue" exists for the dataset
    And a Linear Regression model has been trained and selected
    When the model is deployed
    Then a deployment is created with an active prediction endpoint
    And the endpoint URL follows the pattern "/api/predict/{id}"
    And the deployment has a public dashboard URL

  Scenario: Making a prediction returns a numeric forecast
    Given a feature set targeting "revenue" exists for the dataset
    And a Linear Regression model has been trained, selected, and deployed
    When a prediction is submitted with feature values matching the training schema
    Then the response contains a numeric prediction
    And the response includes the input features that were used

  Scenario: Batch prediction on uploaded CSV returns enriched output
    Given a feature set targeting "revenue" exists for the dataset
    And a Linear Regression model has been trained, selected, and deployed
    When a batch prediction CSV is submitted with 3 rows
    Then the response CSV has a "revenue_prediction" column
    And the output has 3 rows matching the input
