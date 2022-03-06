# Implementing-And-Comparing-Recommender-Systens
implement various recommender system models and compare their performance. 
1. [General](#General)
2. [Program Strcuture](#Program Structure)
3. [Installation](#Installation)

4. [Usage](#Usage)
5. [Footnote](#footnote)
## General
The goal is to predict user rating based on recommender systems.The file of ratings our learning procces was based on includs 100005 records with the following features:
* user – The user’s unique identifier 
* item – The item’s unique identifier 
* rating – The rating that was given to the item by the user, it is in the range [0.5,5] 
* timestamp – The timestamp in which the rating was given. 


## Program Structure
The file learners includs the implention of the diffrent learners:
* baseline model - 𝑟̂𝑢𝑖 = 𝑅̂ + 𝑏𝑢 + 𝑏𝑖 where 𝑅̂ 
is the average of all the ratings in the user-item ratings matrix 𝑅, 𝑏𝑢 is the average rating deviation for user 𝑢 
and 𝑏𝑖 is the average rating deviation for item 𝑖. 
* Neighborhood Recommender - based on 3 nearest neighbors.
* LS Recommender - Uses regression model to predict the ratings
![image](https://i.imgur.com/9qgUOjF.png)
