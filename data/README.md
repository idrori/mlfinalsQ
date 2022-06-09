This folder contains json files representing every question evaluated in this paper. The files are organized by directories named after the semester the final was given. Each json contains the following fields:

| Field | Description |
| ----- | ----------- |
| "Semester" | The semester the question's final was given in (ex. Fall 2017)|
| "Question Number" | The number of the question from the final (ex. 1, 2...) |
| "Sub-Question Number" | The sub-question label the question has (ex. a, a.i) |
| "Points" | The number of points the question is worth |
| "Topic" | The primary machine learning topic that the question targets. The possible topics were regression, classifiers, logistic regression, features, loss functions, neural networks, CNNs, MDPs, RNNs, reinforcement learning, clustering, and decision trees |
| "Type" | Text if the question only relies on text, Image if the question relies on an image |
| "Question" | The original question text as presented from the source |
| "Solution" | The solution to the question |
