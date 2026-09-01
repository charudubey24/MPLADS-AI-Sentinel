import numpy as np
from sklearn.ensemble import IsolationForest


# ==========================================
# MPLADS AI SENTINEL
# AI ANOMALY DETECTION MODEL
# ==========================================


# Sample normal project data
training_data = np.array([
    [3000000, 1500000, 45, 50, 90, 180, 3],
    [5000000, 2500000, 50, 50, 100, 200, 4],
    [4000000, 1800000, 40, 45, 80, 180, 3],
    [2500000, 1000000, 55, 40, 70, 150, 2],
    [6000000, 3200000, 55, 53, 120, 220, 4],
    [3500000, 1700000, 48, 49, 90, 180, 3],
    [4500000, 2200000, 52, 49, 100, 200, 3],
    [7000000, 3500000, 50, 50, 110, 220, 4],
    [3000000, 1400000, 60, 47, 80, 160, 3],
    [5500000, 2700000, 52, 49, 100, 200, 3],
    [4200000, 2100000, 47, 50, 95, 190, 3],
    [3800000, 1900000, 51, 50, 90, 180, 3],
    [6500000, 3300000, 56, 51, 120, 240, 4],
    [2800000, 1300000, 54, 46, 75, 160, 2],
    [4800000, 2400000, 50, 50, 100, 200, 3],
    [5200000, 2600000, 49, 50, 105, 210, 3],
    [3200000, 1600000, 52, 50, 80, 160, 2],
    [7500000, 3800000, 51, 51, 130, 250, 4],
    [4300000, 2100000, 48, 49, 95, 190, 3],
    [3600000, 1800000, 50, 50, 85, 170, 2]
])


# Create AI model
model = IsolationForest(
    n_estimators=200,
    contamination=0.10,
    random_state=42
)


# Train AI model
model.fit(training_data)

print("AI MODEL TRAINED SUCCESSFULLY")


def analyze_project(project):

    # --------------------------------------
    # Read project information
    # --------------------------------------

    budget = float(project["budget"])
    spent = float(project["spent"])

    physical_progress = float(
        project["physical_progress"]
    )

    days_elapsed = float(
        project["days_elapsed"]
    )

    expected_duration = float(
        project["expected_duration"]
    )

    vendors = float(
        project["vendors"]
    )


    # --------------------------------------
    # Calculate financial progress
    # --------------------------------------

    if budget > 0:

        financial_progress = (
            spent / budget
        ) * 100

    else:

        financial_progress = 0


    # --------------------------------------
    # Prepare AI features
    # --------------------------------------

    features = np.array([[
        budget,
        spent,
        physical_progress,
        financial_progress,
        days_elapsed,
        expected_duration,
        vendors
    ]])


    # --------------------------------------
    # AI anomaly prediction
    # --------------------------------------

    prediction = model.predict(features)[0]

    raw_score = model.decision_function(
        features
    )[0]


    # --------------------------------------
    # Risk score
    # --------------------------------------

    risk_score = int(
        max(
            0,
            min(
                100,
                (0.5 - raw_score) * 100
            )
        )
    )


    # --------------------------------------
    # Reasons for anomaly
    # --------------------------------------

    reasons = []


    difference = (
        financial_progress -
        physical_progress
    )


    if difference > 30:

        reasons.append(
            "Financial progress is significantly higher than physical progress."
        )


    if financial_progress > 80:

        reasons.append(
            "More than 80% of the project budget has been utilized."
        )


    if (
        days_elapsed > 0
        and financial_progress > 70
        and days_elapsed < expected_duration * 0.30
    ):

        reasons.append(
            "Unusually high expenditure detected at an early project stage."
        )


    if vendors <= 1:

        reasons.append(
            "Project depends on a single vendor."
        )


    if physical_progress < 30:

        reasons.append(
            "Physical progress is below 30%."
        )


    if prediction == -1:

        reasons.append(
            "Machine-learning model identified the project as statistically unusual."
        )


    if len(reasons) == 0:

        reasons.append(
            "No significant anomaly detected."
        )


    # --------------------------------------
    # Risk level
    # --------------------------------------

    if risk_score >= 70:

        risk_level = "HIGH"

    elif risk_score >= 40:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    # --------------------------------------
    # Return clean JSON-compatible result
    # --------------------------------------

    return {

        "risk_score": int(risk_score),

        "risk_level": str(risk_level),

        "financial_progress": float(
            round(financial_progress, 2)
        ),

        "physical_progress": float(
            round(physical_progress, 2)
        ),

        "anomaly": bool(
            prediction == -1
        ),

        "reasons": [
            str(reason)
            for reason in reasons
        ]

    }


# ==========================================
# TEST MODEL
# ==========================================

if __name__ == "__main__":

    test_project = {

        "budget": 5000000,

        "spent": 4200000,

        "physical_progress": 25,

        "days_elapsed": 30,

        "expected_duration": 180,

        "vendors": 1

    }


    result = analyze_project(
        test_project
    )


    print()
    print("==============================")
    print("AI ANALYSIS RESULT")
    print("==============================")

    print(
        "Risk Score:",
        result["risk_score"]
    )

    print(
        "Risk Level:",
        result["risk_level"]
    )

    print(
        "Financial Progress:",
        result["financial_progress"],
        "%"
    )

    print(
        "Physical Progress:",
        result["physical_progress"],
        "%"
    )

    print(
        "Anomaly:",
        result["anomaly"]
    )

    print()
    print("Reasons:")

    for reason in result["reasons"]:

        print("-", reason)