"""Ham tien ich du doan cluster tu mot pipeline sklearn da huan luyen."""

def predict_segment(model, gender, age, annual_income, spending_score):
    import pandas as pd
    sample = pd.DataFrame([{
        'Gender': gender,
        'Age': age,
        'Annual Income (k$)': annual_income,
        'Spending Score (1-100)': spending_score,
    }])
    return int(model.predict(sample)[0])
