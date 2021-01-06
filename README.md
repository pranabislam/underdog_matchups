### NBA Underdog Betting Project 

This project (work in progress) aims to create a model to capitalize on mispriced underdog odds in NBA matchups to turn a profit in the betting market on an otherwise rarely-explored niche. Underdogs win around one-third of NBA matchups, providing a potentially lucrative opportunity. 

The historical odds scraped as part of this project are courtesy of betexplorer.com, while the API used to scrape basketball-reference.com for statistics comes from https://github.com/vishaalagartha/basketball_reference_scraper. 

We train three different models (logistic regression, decision trees, support vector machines) on historical data dating back to 2009. 

In the API_fixes folder, we make minor adjustments to the basketball-reference-scraper to suit our needs.
In the bref_feature_engineering folder, a number of functions to consolidate stats and odds data into final merged dataframes for each season are included.
In the model_selection folder are functions corresponding to each model type which was trained.
The scraping folder includes preliminary work done to pull data from our respective data sources.

A final report of results and observations is also included
