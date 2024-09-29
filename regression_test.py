import numpy as np
from pytest import approx

from IPython.display import display, HTML


from IPython.display import HTML, display


# Define the HTML and CSS for the green info box with a smiley
message = """
<div style="
    padding: 10px;
    border-radius: 5px;
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    font-size: 16px;
    font-family: Arial, sans-serif;
    margin: 10px 0;
">
    <span style="font-size: 20px;">&#128512;</span>  <!-- Smiley emoji -->
    <strong>Gut gemacht!</strong> <!-- Text -->
</div>
"""

def display_message(msg):
    message = """
<div style="
    padding: 10px;
    border-radius: 5px;
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    font-size: 16px;
    font-family: Arial, sans-serif;
    margin: 10px 0;
">
    <span style="font-size: 20px;">&#128512;</span>  <!-- Smiley emoji -->
    <strong>%s</strong> <!-- Text -->
</div>
    """ % msg
    display(HTML(message))

def check_Regressor(locals):
    assert 'Regressor' in locals, '"Regressor" Klasse nicht gefunden'
    Regressor = locals['Regressor']
    
    reg = Regressor()
    assert hasattr(reg,'fit'), 'Regressor hat keine fit-Funktion'
    assert callable(reg.fit), 'Regressor hat keine fit-Funktion'
    return reg 

def test_regression_parameters(locals):
    reg = check_Regressor(locals)
    data_X = np.array([[0],[0.5],[1]])
    data_y = np.array([0.5,1.5,2.5])
    reg.fit(data_X, data_y)
    assert hasattr(reg,'beta'), 'Regressor hat kein beta-Attribut (zur Speicherung der Gewichte)'
    assert(np.abs(reg.beta[0]-0.5)<1e-5), "Regressor hat falsche beta-Werte."
    assert(np.abs(reg.beta[1]-2.0)<1e-5), "Regressor hat falsche beta-Werte."
    print('Regressions-Parameter OK')
    display(HTML(message))


def test_regression_inference(locals):
    data_X = np.array([[0],[0.5],[1]])
    data_y = np.array([0.5,1.5,2.5])
    reg = check_Regressor(locals)
    reg.fit(data_X, data_y)

    new_data_X = np.array([[0.25], [0.5]])

    assert(np.abs(reg.predict(new_data_X)[0]- 1.0)<1e-5), 'Regression liefert falsche Ergebnisse'
    assert(np.abs(reg.predict(new_data_X)[1]- 1.5)<1e-5), 'Regression liefert falsche Ergebnisse'
    print('Regressions Inferenz OK')
    display(HTML(message))



def test_regression_notebook_datasplit(locals):
    # Test data split
    for reqarray in ['X_train','X_test','y_train','y_test']:
        assert reqarray in locals, "Variable "+str(reqarray)+' nicht vorhanden.'
        assert isinstance(locals[reqarray], np.ndarray)
        if ('train' in reqarray):
            assert(locals[reqarray].shape[0]==60), str(reqarray)+' hat falsche Größe'
        else:
            assert(locals[reqarray].shape[0]==40), str(reqarray)+' hat falsche Größe'

    display(HTML(message))

def test_regression_eval(locals):
    pred=locals['pred']
    y_test = locals['y_test']
    assert 'pred' in locals, 'Prediction nicht in Variable "pred" gefunden.'
    assert 'y_test' in locals, 'Test-Labels sollten in "y_test" vorhanden sein.'
    # Proper length? 
    assert pred.shape[0]==40, "Prediction hat falsche Länge (sollte 40 sein)"
    # Check if prediction is approximately target
    assert np.mean(pred) == approx(np.mean(y_test),0.05), "Prediction ist nicht nah an Labels - Regression hat nicht (gut) funktioniert."
    display(HTML(message))


def test_regression_mse(locals):
    # Test existance and proper value of MSE
    assert ('mse' in locals), "Variable 'mse' nicht definiert"

    assert(locals['mse'] < 0.05), 'MSE zu groß (erwartet: <0.05).'
    display(HTML(message))
