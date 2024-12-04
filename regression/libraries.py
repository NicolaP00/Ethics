import matplotlib.pyplot as plt
import numpy as np
import shap 

def create_explanations(model, X_preprocessed, mlModel):

    # Add feature names
    if mlModel=='rf' or mlModel=='dt':
        explainer = shap.TreeExplainer(model['regressor'].best_estimator_, X_preprocessed, check_additivity=False)
    else:
        explainer = shap.Explainer(model['regressor'].best_estimator_, X_preprocessed, check_additivity=False)
    explanations = explainer(X_preprocessed, check_additivity=False)
    return explanations

def summaryPlot(model, X_preprocessed, lf, output_folder, save_name, plot_type, mlModel):
    explanations = create_explanations(model, X_preprocessed, mlModel)

    # Create plot 
    fig, ax = plt.subplots()
    shap.summary_plot(explanations, X_preprocessed, feature_names=lf, show=False, auto_size_plot=True, plot_type=plot_type, max_display=len(lf), sort=False)
    plt.tight_layout()
    fig.savefig(output_folder + save_name)
    plt.close()

def HeatMap_plot(model, X_preprocessed, output_folder, save_name, lf, mlModel):
    explanations = create_explanations(model, X_preprocessed, mlModel)
    explanations.feature_names = [el for el in lf]

    # Create plot
    fig, ax = plt.subplots()
    shap.plots.heatmap(explanations, max_display=len(lf), show=False, plot_width=22)
    plt.tight_layout()
    plt.title("Features Influence's heatmap")
    fig.savefig(output_folder + save_name)
    plt.close()
    
def Waterfall(model, X_preprocessed, output_folder, num_example, save_name, lf, mlModel):
    explanations = create_explanations(model, X_preprocessed, mlModel)
    explanations.feature_names = [el for el in lf]
    explanation = explanations[num_example, :]

    # Create plot
    fig, ax = plt.subplots() 
    shap.plots.waterfall(explanation, max_display=len(lf), show=False)
    ax.set_title(f"Example {num_example}")
    plt.tight_layout()
    plt.savefig(output_folder + save_name)
    plt.close()

def Decision_plot(model, X_preprocessed, output_folder, num_example, save_name, lf, mlModel):

    # Dataset preprocessing
    if mlModel=='rf' or mlModel=='dt':
        explainer = shap.TreeExplainer(model['regressor'].best_estimator_, X_preprocessed, check_additivity=False)
    else:
        explainer = shap.Explainer(model['regressor'].best_estimator_, X_preprocessed, check_additivity=False)
    explanations = explainer(X_preprocessed, check_additivity=False)
    explanations.feature_names = [el for el in lf]
    explanation = explanations.values[num_example, :]

    # Create plot
    fig, ax = plt.subplots() 
    shap.plots.decision(explainer.expected_value, explanation, feature_names = lf,show=False, feature_display_range = range(len(lf)))
    ax.set_title(f"Example {num_example}")
    plt.tight_layout()
    plt.savefig(output_folder + save_name)
    plt.close()
