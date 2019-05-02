# emailinsight
Just a place to store my scripts/code for tinkering with email data. Eventually may make it into more of a visualization/insight tool.

If you want to use this as the starting point for your own experimentation with emails, the files that have most of the logic are kerasClassify.py and kerasExperiment.py. The former has functions related to creating the dataset and the classifiers, and the later has functions for running tests and reporting results. 

## Getting Started 
run kerasExperiment.py and observe results with the Sample dataset (below). It should give Test accuracy ~ 0.8

## Dataset 
The default dataset is a small subset of Enron emails dataset - just a single user with 6 folders. See parsing and format documentation in parseEmailsCSV
You can export your gmail archive. Comment the call to parseEmailsCSV in get_emails function in kerasClassify.py and Uncomment the parseEmails() call.
It is hard coded to just look in the current directory, but you can modify it as needed - in general, the code is for you to use an inspiration but is not really more than a collection of scripts I wrote up.
