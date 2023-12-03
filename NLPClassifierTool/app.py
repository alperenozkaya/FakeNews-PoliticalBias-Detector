# Model-Extension Integration

import fakenews_prediction


text = "SHARE THIS EVERYWHERE! DISEASED REFUGEES Get SSN and Passport Upon Arrival to the U.S. [Video]"
label, probability = fakenews_prediction.predict_label(text)
print(f"Predicted Label: {label}, Probability: {probability}")

