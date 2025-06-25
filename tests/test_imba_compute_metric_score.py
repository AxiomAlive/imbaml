import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from main import Imba

@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    return pd.DataFrame(X), pd.Series(y)

def test_compute_metric_score_f1(sample_data):
    X, y = sample_data
    hyper_parameters = {'model_class': 'XGBClassifier', 'max_depth': 3, 'n_estimators': 10}
    imba = Imba(metric='f1')
    result = imba.compute_metric_score(hyper_parameters, imba._metric, X, y)
    assert 'loss' in result
    assert result['status'] == 'ok'

def test_compute_metric_score_balanced_accuracy(sample_data):
    X, y = sample_data
    hyper_parameters = {'model_class': 'XGBClassifier', 'max_depth': 3, 'n_estimators': 10}
    imba = Imba(metric='balanced_accuracy')
    result = imba.compute_metric_score(hyper_parameters, imba._metric, X, y)
    assert 'loss' in result
    assert result['status'] == 'ok'

def test_compute_metric_score_invalid_model(sample_data):
    X, y = sample_data
    hyper_parameters = {'model_class': 'InvalidModel', 'max_depth': 3, 'n_estimators': 10}
    imba = Imba(metric='f1')
    with pytest.raises(ValueError):
        imba.compute_metric_score(hyper_parameters, imba._metric, X, y)import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from imbaml.main import Imba

class TestImba(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(n_samples=100, n_features=20, random_state=42)
        self.imba = Imba(metric='f1', n_evals=10)

    def test_compute_metric_score_f1(self):
        hyper_parameters = {
            'model_class': 'SomeClassifier',  # Replace with an actual classifier
            'param1': 1,  # Replace with actual parameters
            'param2': 2
        }
        result = self.imba.compute_metric_score(hyper_parameters, self.imba._metric, self.X, self.y)
        self.assertIn('loss', result)
        self.assertIn('status', result)
        self.assertEqual(result['status'], STATUS_OK)

    def test_compute_metric_score_invalid_model(self):
        hyper_parameters = {
            'model_class': 'InvalidClassifier',  # Invalid classifier
            'param1': 1,
            'param2': 2
        }
        with self.assertRaises(ValueError):
            self.imba.compute_metric_score(hyper_parameters, self.imba._metric, self.X, self.y)

if __name__ == '__main__':
    unittest.main()