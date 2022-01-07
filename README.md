# 02476_ml_ops_project
Progress based on [project checklist](https://github.com/SkafteNicki/dtu_mlops/blob/main/projects/projects.md)

## Project description
### Overall goal of the project
The overall goal of the project is to get familiar with the tools introduced in the course and to use these tools on a leaf classification kaggle challenge.  

The focus will lie on keeping the test results reproducible through use of version control and ensuring that data related to hyperparameters and model weights are properly tagged. This will be done through containerizing the machine learning model or a complete description of the environment used, such that it can be deployed automatically e.g., after successful git commits. 

### What framework are you going to use (Kornia, Transformer, Pytorch-Geometrics)
This project will utilize Kornia in order to complete a kaggle image classification challenge. 

### How to you intend to include the framework into your project
Kornia implements differential algorithms unlike torchvision for transforms, which will be utilized for data augmentation. Kornia also includes various models such as VisionTransformer, which can be used for classification. 
### What data are you going to run on (initially, may change)
[Kaggle - Petals to the Metal](https://www.kaggle.com/c/tpu-getting-started/data)

### What deep learning models do you expect to use
VisionTransformer or traditional CNN 

# Checklist
## Week 1

- [x] Create a git repository
- [x] Make sure that all team members have write access to the github repository
- [ ] Create a dedicated environment for you project to keep track of your packages (using conda)
- [ ] Create the initial file structure using cookiecutter
- [ ] Fill out the `make_dataset.py` file such that it downloads whatever data you need and 
- [ ] Add a model file and a training script and get that running
- [ ] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
- [ ] Remember to comply with good coding practices (`pep8`) while doing the project
- [ ] Do a bit of code typing and remember to document essential parts of your code
- [ ] Setup version control for your data or part of your data
- [ ] Construct one or multiple docker files for your code
- [ ] Build the docker files locally and make sure they work as intended
- [ ] Write one or multiple configurations files for your experiments
- [ ] Used Hydra to load the configurations and manage your hyperparameters
- [ ] When you have something that works somewhat, remember at some point to to some profiling and see if you can optimize your code
- [ ] Use wandb to log training progress and other important metrics/artifacts in your code
- [ ] Use pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

## Week 2

- [ ] Write unit tests related to the data part of your code
- [ ] Write unit tests related to model construction
- [ ] Calculate the coverage.
- [ ] Get some continues integration running on the github repository
- [ ] (optional) Create a new project on `gcp` and invite all group members to it
- [ ] Create a data storage on `gcp` for you data
- [ ] Create a trigger workflow for automatically building your docker images
- [ ] Get your model training on `gcp`
- [ ] Play around with distributed data loading
- [ ] (optional) Play around with distributed model training
- [ ] Play around with quantization and compilation for you trained models

## Week 3

- [ ] Deployed your model locally using TorchServe
- [ ] Checked how robust your model is towards data drifting
- [ ] Deployed your model using `gcp`
- [ ] Monitored the system of your deployed model
- [ ] Monitored the performance of your deployed model

## Additional

- [ ] Revisit your initial project description. Did the project turn out as you wanted?
- [ ] Make sure all group members have a understanding about all parts of the project
- [ ] Create a presentation explaining your project
- [ ] Uploaded all your code to github
- [ ] (extra) Implemented pre-commit hooks for your project repository
- [ ] (extra) Used Optuna to run hyperparameter optimization on your model