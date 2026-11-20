# CS551 Final Project
This project solves a problem for PROACT, a research project involving transportation safety conducted (in part) by graduate students at The University of Alabama. The issue is that researchers currently do not have access to pedestrian information (e.g. the count of pedestrians per minute at any given intersection) when building traffic simulations.

The purpose of our project is to create and test a solution which:
- Takes surveillance camera footage from an intersection as input
- Uses Computer Vision to count the number of pedestrians crossing the intersection over every minute of the video
- Outputs a spreadsheet with this information, along with other relevant statistics

To create and verify our approach, we separate the project into 2 parts:
- /human-verification - contains a desktop application used to take a traffic video as input, allow human labelers to manually count pedestrians every minute, and store the count at the end of each minute. Once the video is over, store the results in a .csv file.
- /opencv - contains an OpenCV-powered desktop application which automates the pedestrian-counting task in the /human-verification folder
