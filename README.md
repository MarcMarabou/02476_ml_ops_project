# 02476_ml_ops_project
Progress based on [project checklist](https://github.com/SkafteNicki/dtu_mlops/blob/main/projects/projects.md)

## Project description
### Overall goal of the project
The overall goal of the project is to get familiar with the tools introduced in the course and to use these tools on a leaf classification kaggle challenge.  

The focus will lie on keeping the test results reproducible through use of version control and ensuring that data related to hyperparameters and model weights are properly tagged. This will be done through containerizing the machine learning model or a complete description of the environment used, such that it can be deployed automatically e.g., after successful git commits. 

### What framework are you going to use (Kornia, Transformer, Pytorch-Geometrics)
This project will utilize Kornia in order to complete a kaggle image classification challenge. 

### How to you intend to include the framework into your project
Kornia implements differential algorithms unlike torchvision for transforms, which will be utilized for data augmentation. Kornia also includes various models such as Vision Transformer (ViT), which can be used for classification. 

### What data are you going to run on (initially, may change)
[Kaggle - Petals to the Metal](https://www.kaggle.com/c/tpu-getting-started/data)

### What deep learning models do you expect to use
The [ViT](https://kornia.readthedocs.io/en/latest/models/vit.html) model.

# Checklist
## Week 1

- [x] Create a git repository
- [x] Make sure that all team members have write access to the github repository
- [x] Create a dedicated environment for you project to keep track of your packages (using conda)
- [x] Create the initial file structure using cookiecutter
- [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and 
- [x] Add a model file and a training script and get that running
- [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
- [x] Remember to comply with good coding practices (`pep8`) while doing the project
- [x] Do a bit of code typing and remember to document essential parts of your code
- [x] Setup version control for your data or part of your data
- [x] Construct one or multiple docker files for your code
- [x] Build the docker files locally and make sure they work as intended
- [ ] Write one or multiple configurations files for your experiments
- [ ] Used Hydra to load the configurations and manage your hyperparameters
- [ ] When you have something that works somewhat, remember at some point to to some profiling and see if you can optimize your code
- [x] Use wandb to log training progress and other important metrics/artifacts in your code
- [x] Use pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

## Week 2

- [x] Write unit tests related to the data part of your code
- [x] Write unit tests related to model construction
- [ ] Calculate the coverage.
- [x] Get some continues integration running on the github repository
- [x] (optional) Create a new project on `gcp` and invite all group members to it
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

A short description of the project.

## References

```
@article{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={ICLR},
  year={2021}
}

@article{steiner2021augreg,
  title={How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers},
  author={Steiner, Andreas and Kolesnikov, Alexander and and Zhai, Xiaohua and Wightman, Ross and Uszkoreit, Jakob and Beyer, Lucas},
  journal={arXiv preprint arXiv:2106.10270},
  year={2021}
}
```

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
