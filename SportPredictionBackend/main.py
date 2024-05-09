from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the pre-trained RandomForest model from the .pkl file
model = joblib.load('random_forest_model.pkl')

# Initialize FastAPI application
app = FastAPI()

# Define the input data structure (adjust according to your feature columns)
class UserInput(BaseModel):
    height: float
    weight: float
    age: int
    sex: str  # Assume 'male' or 'female'
    bmi: float

# Define a prediction endpoint
@app.post("/predict")
async def predict(input: UserInput):
    # Prepare input data for the model
    sex_numeric = 0 if input.sex.lower() == "male" else 1  # Convert sex to numeric

    input_data = np.array([[input.height, input.weight, input.age, sex_numeric, input.bmi]])

    # Use the loaded model to make predictions
    predicted_class = model.predict(input_data)[0]
    print(predicted_class)
    return {"PredictedSport": predicted_class}

# If running the file directly, start a development server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)