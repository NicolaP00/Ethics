# Ethics
## utils

Contiene il materiale utilizzato in progetti precedenti:
- main.py contiene alcuni metodi classici di explainability, come LIME, Shap e DiCE. Viene utilizzata anche la Anova decomposition, le cui funzioni sono scritte in libraries_anova.py.
- FoolingAI.py utilizza un Polynomial Regressor fittato utilizzando delle loss function che incentivano l'utilizzo di determinate features a scapito di altre (Adversarial Techniques)
- Adversarial.py contiene un modello che inganna i metodi di explainability classici, come LIME. Alcune features sono più importanti di altre, ma LIME non se ne accorge.
