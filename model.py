import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# MPLADS AI SENTINEL
# HYBRID AI ANOMALY DETECTION ENGINE
# ============================================================

# ------------------------------------------------------------
# NORMAL PROJECT TRAINING DATA
#
# Columns:
# budget
# spent
# physical_progress
# financial_progress
# days_elapsed
# expected_duration
# vendors
# ------------------------------------------------------------

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


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(data):

    budget = data[:, 0]
    spent = data[:, 1]
    physical = data[:, 2]
    financial = data[:, 3]
    days = data[:, 4]
    duration = data[:, 5]
    vendors = data[:, 6]

    # How much more money has been spent compared with
    # actual physical progress.
    progress_gap = financial - physical

    # Percentage of expected project time already consumed.
    time_progress = np.where(
        duration > 0,
        (days / duration) * 100,
        0
    )

    # Difference between time progress and physical progress.
    schedule_gap = time_progress - physical

    # Spending intensity.
    spending_ratio = np.where(
        budget > 0,
        (spent / budget) * 100,
        0
    )

    # Money spent per percentage of physical progress.
    cost_efficiency = np.where(
        physical > 0,
        spending_ratio / physical,
        spending_ratio
    )

    return np.column_stack([
        progress_gap,
        schedule_gap,
        spending_ratio,
        physical,
        financial,
        time_progress,
        vendors,
        cost_efficiency
    ])


# ------------------------------------------------------------
# Scale engineered features
# ------------------------------------------------------------

engineered_training_data = create_features(training_data)

scaler = StandardScaler()

scaled_training_data = scaler.fit_transform(
    engineered_training_data
)


# ============================================================
# ISOLATION FOREST
# ============================================================

model = IsolationForest(
    n_estimators=300,
    contamination=0.10,
    random_state=42
)

model.fit(scaled_training_data)

print("AI MODEL TRAINED SUCCESSFULLY")


# ============================================================
# MAIN AI ANALYSIS FUNCTION
# ============================================================

def analyze_project(project):

    # --------------------------------------------------------
    # READ INPUT
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    budget = max(budget, 0)
    spent = max(spent, 0)

    physical_progress = max(
        0,
        min(physical_progress, 100)
    )

    days_elapsed = max(days_elapsed, 0)
    expected_duration = max(expected_duration, 1)
    vendors = max(vendors, 0)


    # --------------------------------------------------------
    # FINANCIAL PROGRESS
    # --------------------------------------------------------

    if budget > 0:

        financial_progress = (
            spent / budget
        ) * 100

    else:

        financial_progress = 0


    # Don't allow impossible percentages.
    financial_progress = max(
        0,
        min(financial_progress, 100)
    )


    # ========================================================
    # ENGINEERED AI FEATURES
    # ========================================================

    progress_gap = (
        financial_progress -
        physical_progress
    )

    time_progress = (
        days_elapsed /
        expected_duration
    ) * 100

    schedule_gap = (
        time_progress -
        physical_progress
    )

    spending_ratio = financial_progress

    if physical_progress > 0:

        cost_efficiency = (
            spending_ratio /
            physical_progress
        )

    else:

        cost_efficiency = spending_ratio


    # ========================================================
    # MACHINE LEARNING ANOMALY DETECTION
    # ========================================================

    raw_features = np.array([[
        progress_gap,
        schedule_gap,
        spending_ratio,
        physical_progress,
        financial_progress,
        time_progress,
        vendors,
        cost_efficiency
    ]])

    scaled_features = scaler.transform(
        raw_features
    )

    prediction = model.predict(
        scaled_features
    )[0]

    raw_score = model.decision_function(
        scaled_features
    )[0]


    # ========================================================
    # DOMAIN RISK ENGINE
    #
    # This prevents the result from behaving like a fixed
    # calculator while still giving the ML model influence.
    # ========================================================

    domain_risk = 0

    reasons = []


    # --------------------------------------------------------
    # 1. FINANCIAL VS PHYSICAL PROGRESS
    # --------------------------------------------------------

    if progress_gap >= 50:

        domain_risk += 35

        reasons.append(
            "Critical expenditure-to-progress gap detected."
        )

    elif progress_gap >= 35:

        domain_risk += 28

        reasons.append(
            "Financial progress is significantly higher than physical progress."
        )

    elif progress_gap >= 20:

        domain_risk += 18

        reasons.append(
            "Spending is progressing faster than physical work."
        )

    elif progress_gap >= 10:

        domain_risk += 8

        reasons.append(
            "A moderate gap exists between expenditure and physical progress."
        )


    # --------------------------------------------------------
    # 2. HIGH BUDGET UTILIZATION
    # --------------------------------------------------------

    if financial_progress >= 90:

        domain_risk += 25

        reasons.append(
            "More than 90% of the project budget has been utilized."
        )

    elif financial_progress >= 75:

        domain_risk += 18

        reasons.append(
            "A high proportion of the project budget has been utilized."
        )

    elif financial_progress >= 60:

        domain_risk += 8

        reasons.append(
            "Budget utilization has crossed 60%."
        )


    # --------------------------------------------------------
    # 3. LOW PHYSICAL PROGRESS
    # --------------------------------------------------------

    if physical_progress < 20:

        domain_risk += 25

        reasons.append(
            "Physical progress is critically low."
        )

    elif physical_progress < 35:

        domain_risk += 16

        reasons.append(
            "Physical progress is below 35%."
        )

    elif physical_progress < 50:

        domain_risk += 6


    # --------------------------------------------------------
    # 4. TIME / SCHEDULE RISK
    # --------------------------------------------------------

    if time_progress >= 90 and physical_progress < 70:

        domain_risk += 25

        reasons.append(
            "Project is near its expected completion time but physical progress remains low."
        )

    elif time_progress >= 70 and physical_progress < 50:

        domain_risk += 20

        reasons.append(
            "Physical progress is significantly behind the project timeline."
        )

    elif time_progress >= 50 and physical_progress < 30:

        domain_risk += 15

        reasons.append(
            "Project progress is substantially behind the elapsed timeline."
        )


    # --------------------------------------------------------
    # 5. EARLY HIGH EXPENDITURE
    # --------------------------------------------------------

    if (
        days_elapsed > 0
        and financial_progress >= 70
        and time_progress < 30
    ):

        domain_risk += 25

        reasons.append(
            "Unusually high expenditure detected at an early project stage."
        )


    # --------------------------------------------------------
    # 6. VENDOR CONCENTRATION
    # --------------------------------------------------------

    if vendors <= 1:

        domain_risk += 10

        reasons.append(
            "Project depends on a single vendor."
        )

    elif vendors == 2:

        domain_risk += 4


    # ========================================================
    # MACHINE LEARNING CONTRIBUTION
    # ========================================================

    #
    # decision_function:
    #
    # positive  -> more normal
    # negative  -> more anomalous
    #
    # Convert it into a controlled anomaly contribution.
    #

    ml_anomaly_strength = max(
        0,
        min(
            1,
            0.5 - raw_score
        )
    )

    ml_risk = ml_anomaly_strength * 35


    # If Isolation Forest explicitly identifies anomaly,
    # give it additional influence.
    if prediction == -1:

        ml_risk += 10

        reasons.append(
            "Machine-learning model identified the project as statistically unusual."
        )


    # ========================================================
    # FINAL RISK SCORE
    # ========================================================

    risk_score = (
        domain_risk * 0.65
        +
        ml_risk * 0.35
    )


    # --------------------------------------------------------
    # Add a small severity boost for extreme combinations.
    # --------------------------------------------------------

    if (
        financial_progress >= 80
        and physical_progress <= 30
    ):

        risk_score += 15

        if (
            "Critical expenditure-to-progress gap detected."
            not in reasons
        ):

            reasons.append(
                "High expenditure combined with low physical completion indicates elevated financial risk."
            )


    if (
        progress_gap >= 40
        and time_progress >= 70
    ):

        risk_score += 12

        reasons.append(
            "Expenditure, physical progress and project timeline show a significant mismatch."
        )


    # --------------------------------------------------------
    # Special case:
    # Very healthy project should not remain MEDIUM.
    # --------------------------------------------------------

    if (
        financial_progress <= 60
        and
        physical_progress >= financial_progress - 5
        and
        time_progress <= physical_progress + 15
        and
        vendors >= 2
    ):

        risk_score *= 0.45


    # --------------------------------------------------------
    # Clamp score.
    # --------------------------------------------------------

    risk_score = int(
        max(
            0,
            min(
                100,
                round(risk_score)
            )
        )
    )


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if risk_score >= 70:

        risk_level = "HIGH"

    elif risk_score >= 40:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    # ========================================================
    # CLEAN REASONS
    # ========================================================

    if len(reasons) == 0:

        reasons.append(
            "Project indicators are within the expected range."
        )


    # Remove duplicate reasons.
    reasons = list(
        dict.fromkeys(reasons)
    )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "risk_score": int(
            risk_score
        ),

        "risk_level": str(
            risk_level
        ),

        "financial_progress": float(
            round(
                financial_progress,
                2
            )
        ),

        "physical_progress": float(
            round(
                physical_progress,
                2
            )
        ),

        "anomaly": bool(
            prediction == -1
        ),

        "reasons": [
            str(reason)
            for reason in reasons
        ]

    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    test_projects = [

        {
            "name": "Healthy Project",
            "budget": 5000000,
            "spent": 1500000,
            "physical_progress": 35,
            "days_elapsed": 60,
            "expected_duration": 180,
            "vendors": 4
        },

        {
            "name": "Moderate Risk",
            "budget": 5000000,
            "spent": 3000000,
            "physical_progress": 35,
            "days_elapsed": 100,
            "expected_duration": 180,
            "vendors": 3
        },

        {
            "name": "High Risk",
            "budget": 5000000,
            "spent": 4500000,
            "physical_progress": 20,
            "days_elapsed": 140,
            "expected_duration": 180,
            "vendors": 1
        }

    ]


    print()
    print("========================================")
    print("      MPLADS AI SENTINEL TEST")
    print("========================================")


    for project in test_projects:

        print()
        print("----------------------------------------")
        print(
            "PROJECT:",
            project["name"]
        )
        print("----------------------------------------")

        project_copy = dict(project)

        project_copy.pop(
            "name"
        )

        result = analyze_project(
            project_copy
        )

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

        print("Reasons:")

        for reason in result["reasons"]:

            print(
                "-",
                reason
            )