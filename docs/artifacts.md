# Artifacts of 'XSS adversarial example attacks based on deep reinforcement learning: A Replication and Extension study'

All the artifacts are available at [FigShare](https://figshare.com/articles/dataset/Artifacts_of_XSS_adversarial_example_attacks_based_on_deep_reinforcement_learning_A_Replication_and_Extension_study_/27959817).
In FigShare there are the data and the experiments (runs) in two separate zip files.

## Data
Inside the zip file there is a folder named *data*.
Data contains the [FMereani Dataset](https://github.com/fmereani/Cross-Site-Scripting-XSS/blob/master/XSSDataSets/Payloads.csv) (*FMereani.csv*), the one filtered excluding the non-http requests (*filtered.csv*), and the one filtered by the Oracle (*filtered_oracle.csv*).
The csv files are structured with 2 columns. *Payloads* column contains the input string for the Generated Functions, while the column *Class* contains the ground-truth label (Malicious/Benign).

There is also a folder indicating the vocabulary size (10, representing the $10%$ of the most common tokens).
Inside this folder, there are the file representing the vocabulary (*vocabulary.csv*) and the two folders representing the data used to train the detectors and the adversarial agents respectively.
Inside every folder, there are the 3 dataset splits.

## Experiments
Inside the zip file there is a folder named *runs*.
The folder contains the summary of the results (*summary.json*) and one subfolder for every employed detection model, in this case there are 3 folders: *cnn*, *lstm*, *mlp*.

They contain a folder indicating the vocabulary size (10, representing the $10%$ of the most common tokens).

Inside that, there is the foldder representing the run, which contains the configuration, the results and the checkpoint of the model.

Inside, there are two folders representing the adversarial model targeting that detector, trained respectively with and without the usage of the Oracle to select the reward.

10 runs are available for every detection model, and the folder of each run contains the checkpoint, the results and the studies about the ruin rates.