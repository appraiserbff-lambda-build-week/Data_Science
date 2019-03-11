# Appraiser's BFF 

### Describe the Established data source with at least rough data able to be provided on day 1.

Our data source is the Zillow housing dataset on [Kaggle](https://www.kaggle.com/c/zillow-prize-1/data).  We will perform exploratory data analysis on the data, and be able to provide a baseline model that can be used on the web app.

### You can gather information about the data set you'll be working with from the project description.  

This information is included on the Kaggle website.  And we're also familier with this type of data, so hopefully it won't be a problem for us.

### Write a description for what the DS problem is (what uncertainty/prediction are we trying to do here? Sentiment analysis? Why is this a useful solution to a problem?)

We're mainly looking at a Linear Regression problem:  We take a bunch of inputs (X) such as sqft., number of rooms, pool or no pool, ... and try to predict (y) or Price of house.  A unique characteristic of this problem is that while we can figure out how close we got to historical data, there isn't a for us to verify whether our future predictions are accurate in anyway **until** a real estate transaction actually takes place.  So it will be nearly impossible for us to get instant feedback about our predictions.

### A target (e.g. JSON format or such) for output that DS students can deliver to web/other students for them to ingest and use in the app

We've been asked to deliver our results in JSON, but we're still not sure on how will take place since everytime a user looks up the price of a home, our Python-based machine learning model needs to run the predict() function.  I'm having a hard time trying to figure out how to integrate this python-based ML model into a JS frontend and a Java backend.






