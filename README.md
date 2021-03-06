# IFT6758 Repo Template

This template provides you with a skeleton of a Python package that can be installed into your local machine.
This allows you access your code from anywhere on your system if you've activated the environment the package was installed to.
You are encouraged to leverage this package as a skeleton and add all of your reusable code, functions, etc. into relevant modules.
This makes collaboration much easier as the package could be seen as a "single source of truth" to pull data, create visualizations, etc. rather than relying on a jumble of notebooks.
You can still run into trouble if branches are not frequently merged as work progresses, so try to not let your branches diverge too much!

Also included in this repo is an image of the NHL ice rink that you can use in your plots.
It has the correct location of lines, faceoff dots, and length/width ratio as the real NHL rink.
Note that the rink is 200 feet long and 85 feet wide, with the goal line 11 feet from the nearest edge of the rink, and the blue line 75 feet from the nearest edge of the rink.

<p align="center">
<img src="./figures/nhl_rink.png" alt="NHL Rink is 200ft x 85ft." width="400"/>
<p>

The image can be found in [`./figures/nhl_rink.png`](./figures/nhl_rink.png).

## Installation

To install this package, first setup your Python environment by following the instructions in the [Environment](#environments) section.
Once you've setup your environment, you can install this package by running the following command from the root directory of your repository.

    pip install -e .

You should see something similar to the following output:

    > pip install -e .
    Obtaining file:///home/USER/project-template
    Installing collected packages: ift6758
    Running setup.py develop for ift6758
    Successfully installed ift6758-0.1.0


## Environments

The first thing you should setup is your isolated Python environment.
You can manage your environments through either Conda or pip.
Both ways are valid, just make sure you understand the method you choose for your system.
It's best if everyone on your team agrees on the same method, or you will have to maintain both environment files!
Instructions are provided for both methods.

**Note**: If you are having trouble rendering interactive plotly figures and you're using the pip + virtualenv method, try using Conda instead.

### Conda

Conda uses the provided `environment.yml` file.
You can ignore `requirements.txt` if you choose this method.
Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual) installed on your system.
Once installed, open up your terminal (or Anaconda prompt if you're on Windows).
Install the environment from the specified environment file:

    conda env create --file environment.yml
    conda activate ift6758-conda-env

After you install, register the environment so jupyter can see it:

    python -m ipykernel install --user --name=ift6758-conda-env

You should now be able to launch jupyter and see your conda environment:

    jupyter-lab

If you make updates to your conda `environment.yml`, you can use the update command to update your existing environment rather than creating a new one:

    conda env update --file environment.yml

You can create a new environment file using the `create` command:

    conda env export > environment.yml

### Pip + Virtualenv

An alternative to Conda is to use pip and virtualenv to manage your environments.
This may play less nicely with Windows, but works fine on Unix devices.
This method makes use of the `requirements.txt` file; you can disregard the `environment.yml` file if you choose this method.

Ensure you have installed the [virtualenv tool](https://virtualenv.pypa.io/en/latest/installation.html) on your system.
Once installed, create a new virtual environment:

    vitualenv ~/ift6758-venv
    source ~/ift6758-venv/bin/activate

Install the packages from a requirements.txt file:

    pip install -r requirements.txt

As before, register the environment so jupyter can see it:

    python -m ipykernel install --user --name=ift6758-venv

You should now be able to launch jupyter and see your conda environment:

    jupyter-lab

If you want to create a new `requirements.txt` file, you can use `pip freeze`:

    pip freeze > requirements.txt


## NHL Project Tools
This project also requires installation of the NHL Project Tools library.

To install it in your conda environment, simply run `pip install nhl_api_tools/`
in the project directory.


## Creating Datasets

Note: If working from Milestone 1, to update local datasets to Milestone 2 simply run:
```
python src/data/download_data.py --seasons 2015
python src/data/tidy_data.py
python src/data/split_data_for_training.py
python src/data/process_data_for_training.py
```

### Downloading NHL data
You can download the raw NHL data using python:

    python src/data/download_data.py

which will download the data into `data/raw` directory.

### Creating the tidy data

Then you can clean the data by running this script:

    python src/data/tidy_data.py

which will clean the data in the `data/raw` directory into `csv` files containing the game events (currently only shots and goals) into `data/clean` directory.

### Creating the shot map data

You can create the data files required for the shot maps by running this script:

    python src/data/team_data.py

This will read the tidy data in the `data/clean` directory and save the summarized data in the `data/games` directory.
Afterwards, the Jupyter notebook "shot_maps.ipynb" can be used to visualize the data.

## Interactive visualization with Jupyter Lab
To run `notebooks/shot_maps.ipynb` notebook with the interactive visualization of the shots you need to do the following:
1. Install `jupyter-dash`:

    pip install jupyter-dash

2. Install an extension for Jupyter Lab + dash (for more details visit [this tutorial](https://github.com/plotly/jupyter-dash)):

    jupyter lab build

To check that the extension is installed properly, call `jupyter labextension list`.
