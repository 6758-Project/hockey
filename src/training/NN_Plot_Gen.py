""" Calls the generate_figure_mod function and generate the figures for NN
"""

from generate_figures_mod import generate_adv_model_figures 

title = "Visual Summary - Neural Network Models Comparison"
image_dir = "./figures/exploration/"
experiment_prediction_filenames = {
    "Neural Network - Distance": "./models/predictions/NN_distance.csv",
    "Neural Network - Baseline":"./models/predictions/NN_baseline.csv",
    "Neural Network - Basic Features":"./models/predictions/NN_basic.csv",
    "Neural Network - Advance Features":"./models/predictions/NN_adv.csv"
}
generate_adv_model_figures(experiment_prediction_filenames,title,image_dir)





