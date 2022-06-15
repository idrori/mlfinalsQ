This folder contains the full dataset of final questions evaluated in this paper as both csv files and json files. The files in the jsons directory are organized by subdirectories named after the semester the final was given, and the csvs directory contains csv files of the final questions organized by semester and one file with all finals combined. Each json contains the following fields:

| Field | Description |
| ----- | ----------- |
| "Semester" | The semester the question's final was given in (ex. Fall 2017)|
| "Question Number" | The number of the question from the final (ex. 1, 2...) |
| "Part Number" | The sub-question label the question has (ex. a, b.i) |
| "Points" | The number of points the question is worth |
| "Topic" | The primary machine learning topic that the question targets. The possible topics were regression, classifiers, logistic regression, features, loss functions, neural networks, CNNs, MDPs, RNNs, reinforcement learning, clustering, and decision trees |
| "Type" | Text if the question only relies on text, Image if the question relies on an image |
| "Question" | The original question text as presented from the source |
| "Solution" | The solution to the question |

In the csv files, the above fields correspond to the column headers.

The images directory contains all the images which are part of questions in the finals, with the file name specifying which question the image is required for.

The embeddings directory contains, by semester, the embeddings (given by OpenAI's text-similarity-babbage-001 engine) of questions.
