import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load the pre-trained model and scaler
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the Excel file with the specified engine
df = pd.read_excel('D://Excel/j2.xlsx', engine='openpyxl')

# Define the FastAPI app
app = FastAPI()

# Define the input data model
class InputData(BaseModel):
    entity_id: int
    lang_id: int

# Define the response data model
class PredictionResponse(BaseModel):
    certificate: int
    predicted_rate: float

# Define the prediction endpoint
@app.post("/predict/", response_model=list[PredictionResponse])
def predict(data: InputData):
    certifications = df['certificate'].unique().tolist()
    results = []
    for cert in certifications:
        input_data = pd.DataFrame({
            'entity_id': [data.entity_id],
            'certificate': [cert],
            'lang_id': [data.lang_id]
        })
        input_data_scaled = scaler.transform(input_data)
        prediction = best_model.predict(input_data_scaled)
        results.append(PredictionResponse(certificate=cert, predicted_rate=prediction[0]))
    return results

# Example usage
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
