import pandas as pd

def preprocess_survey_data(df):
    # Mapping for Screen Time / Social Media
    time_map = {
        "Less than 1 hour": 0.5, "1-2 hours": 1.5, "2-4 hours": 3, 
        "4-6 hours": 5, "More than 6 hours": 8,
        "Less than 30 minutes": 0.25, "30–60 minutes": 0.75,
        "1–2 hours": 1.5, "2–3 hours": 2.5, "More than 3 hours": 4
    }
    
    # Mapping for Frequency / Agreement
    freq_map = {
        "Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4, "Very Often": 5,
        "Strongly Disagree": 1, "Disagree": 2, "Neutral": 3, "Agree": 4, "Strongly Agree": 5
    }

    # Apply mappings to your columns (ensure column names match your CSV exactly)
    df['ScreenTime_Numeric'] = df['Total Screen Time Today'].map(time_map)
    df['SocialMedia_Numeric'] = df['Social Media Usage Today'].map(time_map)
    # ... repeat for other columns ...
    
    return df.dropna()