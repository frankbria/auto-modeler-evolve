Feature: CSV Data Upload
  Business analysts upload their data files to start analysis

  Scenario: Upload a valid CSV file
    Given a project exists
    When I upload a CSV file with sales data
    Then the dataset is created with correct row and column counts
    And the preview contains the first rows of data
    And each column has statistics

  Scenario: Upload an invalid file type
    Given a project exists
    When I upload a non-CSV file
    Then the upload is rejected with an error

  Scenario: Preview an uploaded dataset
    Given a project with an uploaded CSV dataset
    When I request the dataset preview
    Then I receive the column statistics and sample rows
