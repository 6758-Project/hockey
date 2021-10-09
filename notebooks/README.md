# Running interactive notebooks with Jupyter Lab


In order to run the "shot_maps.ipynb" on Jupyter Lab you need:

1. Download the NHL raw data using `..\src\data\download_data.py` (e.g. it would be downloaded in `..\data\raw`)
2. Get/save the teams event data from the raw NHL data that was downloaded in `..\data\raw` directory.
3. Install `jupyter-dash`: 
    
    pip install jupyter-dash
    
4. Install an extension for Jupyter Lab + dash (for more details visit [this tutorial](https://github.com/plotly/jupyter-dash)):
    
    jupyter lab build
    
To check that the extension is installed properly, call `jupyter labextension list`.


