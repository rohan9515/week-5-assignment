import pandas as pd
from pycaret.classification import predict_model, load_model


def load_data(filepath):
    """
    Loads churn data into a DataFrame from a string filepath.
    """
    df = pd.read_csv(filepath, index_col='customerID')
    return df


def make_predictions(df, threshold=0.7):
    model = load_model('gbc')
    predictions = predict_model(model, data= df)
    
    predictions.rename({'prediction_label':'predicted_churn'}, axis=1, inplace = True)
    predictions['predicted_churn']=(predictions['prediction_score']>=threshold)
    predictions['predicted_churn'].replace({False:1,True:0}, inplace = True)
    return predictions['predicted_churn']



if __name__== "__main__":
    df = load_data(r"C:\Users\anupu\Downloads\new_churn_data.csv")
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)