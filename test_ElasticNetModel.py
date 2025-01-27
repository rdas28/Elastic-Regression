import numpy
import csv
from ..models.ElasticNet import ElasticNetModel
from ..models.ElasticNet import ElasticNetModelResults

def test_predict():
    data = []
    with open("elasticnet/tests/small_test.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = numpy.array([[v for k,v in datum.items() if k.startswith('x')] for datum in data])
    y = numpy.array([[v for k,v in datum.items() if k=='y'] for datum in data])
    X = X.astype(float)
    y = y.astype(float).flatten() 
    # Data is beeing split into training and testing data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Hyperparameter Optimization through Cross-Validation
    Cross_validation_score_best = -numpy.inf
    leading_parameters = {}

    kcf = 5
    segment_length = len(X_train) // kcf

    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
            validation_scores = []
            for i in range(kcf):
                X_train_segment = numpy.concatenate((X_train[:i*segment_length], X_train[(i+1)*segment_length:]), axis=0)
                y_train_segment = numpy.concatenate((y_train[:i*segment_length], y_train[(i+1)*segment_length:]), axis=0)
                X_validation_subset = X_train[i*segment_length:(i+1)*segment_length]
                y_validation_subset = y_train[i*segment_length:(i+1)*segment_length]
                
                temp_model = ElasticNetModel(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, convergence_criteria=1e-4, step_size=0.005, bias_term=True)
                temp_model.fit(X_train_segment, y_train_segment)
                predicted_y_values = temp_model.predict(X_validation_subset)
                model_results = ElasticNetModelResults(temp_model)
                validation_scores.append(model_results.r2_score(y_validation_subset, predicted_y_values))
            
            mean_evaluation = numpy.mean(validation_scores)
            if mean_evaluation > Cross_validation_score_best:
                Cross_validation_score_best = mean_evaluation
                leading_parameters = {'alpha': alpha, 'l1_ratio': l1_ratio}

    # Display the optimal results of the model
    print("--- Optimal Model Performance Metrics ---")
    print(f"Optimal R² Value Achieved Through Cross-Validation: {Cross_validation_score_best:.4f}")
    print(f"Optimal Alpha Value for Model Performance: {leading_parameters['alpha']}")
    print(f"Optimal L1 Ratio Value for Model Performance: {leading_parameters['l1_ratio']}")

    # Build the Final Model Using Optimal Configuration Settings
    final_model = ElasticNetModel(max_iter=2000, convergence_criteria=1e-4, step_size=0.005, alpha=leading_parameters['alpha'], l1_ratio=leading_parameters['l1_ratio'], bias_term=True)
    results = final_model.fit(X_train, y_train)

    # Generating predictions for the testing data set
    y_pred_test = results.predict(X_test)
    # Setting Up the Model for Performance Evaluation
    result_model = ElasticNetModelResults(final_model)

    # Computing and Presenting Evaluation Metrics
    print("----------------------------------------------")
    print(f"--- Performance Evaluation Summary on the Test Data Set---")
    print(f"R² Score: {result_model.r2_score(y_test, y_pred_test):.4f}")
    print(f"RMSE: {result_model.rmse(y_test, y_pred_test):.4f}")

test_predict()