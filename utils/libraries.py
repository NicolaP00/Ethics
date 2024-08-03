import matplotlib.pyplot as plt
import shap 
import numpy as np
import os

def create_explanations(model, X, preprocessor):
    X_preprocessed = preprocessor.fit_transform(X)

    # Add feature names
    explainer = shap.Explainer(model['regressor'].best_estimator_, X_preprocessed)
    explanations = explainer(X_preprocessed)
    return explanations, X_preprocessed

def summaryPlot(model, X, preprocessor, lf, output_folder, save_name, plot_type):
    explanations, X_preprocessed = create_explanations(model, X, preprocessor)

    # Create plot 
    fig, ax = plt.subplots()
    shap.summary_plot(explanations, X_preprocessed, feature_names=lf, show=False, auto_size_plot=True, plot_type=plot_type, max_display=len(lf), sort=False)
    plt.tight_layout()
    fig.savefig(output_folder + save_name)
    plt.close()

def HeatMap_plot(model, X, preprocessor, output_folder, save_name, lf):
    explanations, _ = create_explanations(model, X, preprocessor)
    explanations.feature_names = [el for el in lf]

    # Create plot
    fig, ax = plt.subplots()
    shap.plots.heatmap(explanations, max_display=len(lf), show=False, plot_width=22)
    plt.tight_layout()
    plt.title("Features Influence's heatmap")
    fig.savefig(output_folder + save_name)
    plt.close()
    
def Waterfall(model, X, preprocessor, output_folder, num_example, save_name, lf):
    explanations, _ = create_explanations(model, X, preprocessor)
    explanations.feature_names = [el for el in lf]
    explanation = explanations[num_example, :]

    # Create plot
    fig, ax = plt.subplots() 
    shap.plots.waterfall(explanation, max_display=len(lf), show=False)
    ax.set_title(f"Example {num_example}")
    plt.tight_layout()
    plt.savefig(output_folder + save_name)
    plt.close()

def Decision_plot(model, X, preprocessor, output_folder, num_example, save_name, lf):

    # Dataset preprocessing
    X_preprocessed = preprocessor.fit_transform(X)
    explainer = shap.Explainer(model['regressor'].best_estimator_, X_preprocessed)
    explanations = explainer(X_preprocessed)
    explanations.feature_names = [el for el in lf]
    explanation = explanations.values[num_example, :]

    # Create plot
    fig, ax = plt.subplots() 
    shap.plots.decision(explainer.expected_value, explanation, feature_names = lf,show=False, feature_display_range = range(len(lf)))
    ax.set_title(f"Example {num_example}")
    plt.tight_layout()
    plt.savefig(output_folder + save_name)
    plt.close()
