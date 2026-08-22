# ============================================================
# AI COURSE RECOMMENDATION SYSTEM
# Flask + Random Forest
# ============================================================


# ============================================================
# 1. LIBRARIES IMPORT
# ============================================================

# Flask website/application banane ke liye use hota hai.
from flask import Flask, render_template, request


# Joblib .pkl files ko load karne ke liye use hota hai.
#
# Hamara trained Random Forest model aur encoders
# .pkl files mein saved hain.
import joblib


# ============================================================
# 2. FLASK APPLICATION CREATE
# ============================================================

# Flask application ka object create kar rahe hain.
#
# __name__ Flask ko current application ka location
# identify karne mein help karta hai.

app = Flask(__name__)


# ============================================================
# 3. TRAINED MODEL LOAD
# ============================================================

# Ye hamara trained Random Forest model hai.
#
# Is model ko train_model.py ne train karke
# course_model.pkl mein save kiya tha.

model = joblib.load(
    "training/model/course_model.pkl"
)


# ============================================================
# 4. ENCODERS LOAD
# ============================================================

# Education encoder
#
# Example:
# FSC / Bachelor's / Master's
# ko numerical values mein convert karega.

education_encoder = joblib.load(
    "training/model/education_encoder.pkl"
)


# Interest encoder
#
# Example:
# Machine Learning
# NLP
# Automation
# etc.
#
# ko numerical values mein convert karega.

interest_encoder = joblib.load(
    "training/model/interest_encoder.pkl"
)


# Career Goal encoder
#
# Example:
# AI Researcher
# ML Engineer
# etc.
#
# ko numerical values mein convert karega.

career_encoder = joblib.load(
    "training/model/career_encoder.pkl"
)


# Course encoder
#
# Model numerical prediction deta hai.
#
# Ye encoder us numerical value ko
# actual course name mein convert karega.

course_encoder = joblib.load(
    "training/model/course_encoder.pkl"
)


# ============================================================
# 5. SKILL LEVEL CONVERSION
# ============================================================

# Hamare training code mein:
#
# python_level
# programming_level
# math_level
# ai_knowledge
#
# already numerical values hain.
#
# Isliye Flask mein user ke selected text ko
# number mein convert karna hoga.


level_mapping = {

    "Beginner": 1,

    "Intermediate": 2,

    "Advanced": 3

}


# ============================================================
# 6. HOME PAGE
# ============================================================

# "/" website ka main/home page hai.
#
# Browser mein:
#
# http://127.0.0.1:5000/
#
# open karne par index.html show hoga.

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# 7. PREDICTION ROUTE
# ============================================================

# Jab student form submit karega,
# request /predict route par jayegi.
#
# POST method form ka data server ko bhejti hai.

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ====================================================
        # 8. STUDENT DATA FORM SE LENA
        # ====================================================

        # Name model ka feature nahi hai.
        #
        # Ye sirf result page par student ka naam
        # show karne ke liye use hoga.

        name = request.form["name"]


        # Age ko integer number mein convert kar rahe hain.

        age = int(
            request.form["age"]
        )


        # Education text form mein milegi.

        education = request.form["education"]


        # Percentage ko decimal number mein convert kar rahe hain.

        percentage = float(
            request.form["percentage"]
        )


        # Student ke Python level ko lena.

        python_level_text = request.form[
            "python_level"
        ]


        # Programming level lena.

        programming_level_text = request.form[
            "programming_level"
        ]


        # Mathematics level lena.

        math_level_text = request.form[
            "math_level"
        ]


        # AI knowledge level lena.

        ai_knowledge_text = request.form[
            "ai_knowledge"
        ]


        # Interest lena.

        interest = request.form["interest"]


        # Career goal lena.

        career_goal = request.form[
            "career_goal"
        ]


        # ====================================================
        # 9. LEVELS KO NUMBERS MEIN CONVERT KARNA
        # ====================================================

        # Example:
        #
        # Beginner = 1
        # Intermediate = 2
        # Advanced = 3

        python_level = level_mapping[
            python_level_text
        ]


        programming_level = level_mapping[
            programming_level_text
        ]


        math_level = level_mapping[
            math_level_text
        ]


        ai_knowledge = level_mapping[
            ai_knowledge_text
        ]


        # ====================================================
        # 10. EDUCATION ENCODE KARNA
        # ====================================================

        # Education text ko numerical value mein
        # convert kar rahe hain.

        education_encoded = (
            education_encoder
            .transform([education])[0]
        )


        # ====================================================
        # 11. INTEREST ENCODE KARNA
        # ====================================================

        # Interest ko numerical value mein convert kar rahe hain.

        interest_encoded = (
            interest_encoder
            .transform([interest])[0]
        )


        # ====================================================
        # 12. CAREER GOAL ENCODE KARNA
        # ====================================================

        # Career goal ko numerical value mein convert kar rahe hain.

        career_encoded = (
            career_encoder
            .transform([career_goal])[0]
        )


        # ====================================================
        # 13. MODEL INPUT
        # ====================================================

        # IMPORTANT:
        #
        # Ye order train_model.py ke X columns
        # ke exactly same hai.
        #
        # 1. age
        # 2. education
        # 3. percentage
        # 4. python_level
        # 5. programming_level
        # 6. math_level
        # 7. ai_knowledge
        # 8. interest
        # 9. career_goal

        features = [[

            age,

            education_encoded,

            percentage,

            python_level,

            programming_level,

            math_level,

            ai_knowledge,

            interest_encoded,

            career_encoded

        ]]


        # ====================================================
        # 14. MACHINE LEARNING PREDICTION
        # ====================================================

        # Ab Random Forest model ko exactly
        # 9 features mil rahe hain.

        prediction = model.predict(
            features
        )


        # ====================================================
        # 15. COURSE NAME MEIN CONVERT
        # ====================================================

        # Model numerical class return karta hai.
        #
        # Course encoder us number ko
        # original course name mein convert karega.

        recommended_course = (
            course_encoder
            .inverse_transform(prediction)[0]
        )


        # ====================================================
        # 16. RESULT PAGE
        # ====================================================

        # Ab result.html ko complete information
        # send kar rahe hain.

        return render_template(

            "result.html",

            name=name,

            age=age,

            education=education,

            percentage=percentage,

            python_level=python_level_text,

            programming_level=programming_level_text,

            math_level=math_level_text,

            ai_knowledge=ai_knowledge_text,

            interest=interest,

            career_goal=career_goal,

            course=recommended_course

        )


    # ========================================================
    # 17. ERROR HANDLING
    # ========================================================

    except Exception as e:

        # Agar koi error aaye to terminal mein
        # actual error print hoga.

        print("\nERROR:")
        print(e)


        # Browser par simple error message show karenge.

        return f"""
        <h2>Prediction Error</h2>

        <p>
            Something went wrong while generating
            the course recommendation.
        </p>

        <p>
            Please check the terminal for details.
        </p>

        <hr>

        <p>
            Error: {e}
        </p>

        <br>

        <a href="/">
            Go Back
        </a>
        """


# ============================================================
# 18. RUN FLASK APPLICATION
# ============================================================

# Ye check karta hai ke app.py directly run hui hai
# ya kisi aur Python file mein import hui hai.

if __name__ == "__main__":

    # Flask development server start hoga.
    #
    # debug=True development ke waqt errors
    # detail mein show karta hai.

    app.run(
        debug=True
    )