import streamlit as st
import numpy as np
import joblib
from PIL import Image
import base64
import pandas as pd


model = joblib.load("knn_model.pkl")


st.set_page_config(page_title="Diabetesguard", page_icon="logo1.png", layout="centered")

language = st.selectbox("🌐 Select language / Оберіть мову", ["English", "Українська"])


T = {
    "intro_title": {
        "English": "Welcome to the Diabetesguard!",
        "Українська": "Ласкаво просимо до Diabetesguard!"
    },
    "intro_text": {
        "English": "This tool helps you assess your <b>risk of developing diabetes</b> based on key medical indicators. By entering values from your recent lab tests — you'll receive an <b>instant prediction</b> powered by machine learning.",
        "Українська": "Цей інструмент допоможе оцінити ваш <b>ризик розвитку цукрового діабету</b> на основі медичних показників. Ввівши актуальні аналізи, ви отримаєте <b>миттєвий прогноз</b> завдяки моделі машинного навчання."
    },
    "goal_list": {
        "English": [
            "Understand your current health status",
            "Identify early warning signs of diabetes",
            "Make informed decisions about visiting a doctor"
        ],
        "Українська": [
            "Зрозуміти свій поточний стан здоров'я",
            "Виявити ранні ознаки діабету",
            "Приймати обґрунтовані рішення щодо візиту до лікаря"
        ]
    },
    "disclaimer": {
        "English": "Please note: This app does not provide a medical diagnosis. Always consult a healthcare professional for clinical decisions.",
        "Українська": "Увага: застосунок не є медичним діагностичним засобом. Завжди консультуйтесь з лікарем для прийняття клінічних рішень."
    },
    "enter_prompt": {
        "English": "Please enter your lab test results below to get a prediction.",
        "Українська": "Введіть свої результати аналізів, щоб отримати прогноз."
    },
    "form_labels": {
        "English": {
            "gender": "Gender",
            "age": "Age",
            "urea": "Urea (mmol/L)",
            "creatinine": "Creatinine (mmol/L)",
            "hba1c": "Glycated hemoglobin (%)",
            "bmi": "BMI (kg/m²)",
            "chol": "Cholesterol (mmol/L)",
            "tg": "Triglycerides (mmol/L)",
            "hdl": "High-density Lipoprotein (mmol/L)",
            "ldl": "Low-density Lipoprotein (mmol/L)",
            "vldl": "Very Low-density Lipoprotein (mmol/L)"
        },
        "Українська": {
            "gender": "Стать",
            "age": "Вік",
            "urea": "Сечовина (ммоль/л)",
            "creatinine": "Креатинін (ммоль/л)",
            "hba1c": "Глікований гемоглобін (%)",
            "bmi": "ІМТ (кг/м²)",
            "chol": "Холестерин (ммоль/л)",
            "tg": "Тригліцериди (ммоль/л)",
            "hdl": "Ліпопротеїди високої щільності (ммоль/л)",
            "ldl": "Ліпопротеїди низької щільності (ммоль/л)",
            "vldl": "Ліпопротеїди дуже низької щільності (ммоль/л)"
        }
    },
    "predict_button": {
    "English": "🔍 Predict",
    "Українська": "🔍 Передбачити"
    },
    "zero_warning": {
    "English": "⚠️ Please ensure that none of the required fields have a value of 0:\nUrea, Creatinine, Glycated Hemoglobin, High-density Lipoprotein, or BMI.",
    "Українська": "⚠️ Будь ласка, переконайтесь, що жодне з обов'язкових полів не має значення 0:\nСечовина, Креатинін, Глікований гемоглобін, ЛПВЩ або ІМТ."
    },
    "result_high_title": {
    "English": "⚠️ High Diabetes Risk",
    "Українська": "⚠️ Високий ризик діабету"
    },
    "result_high_text": {
        "English": "Your estimated risk of diabetes is <strong>{percentage}%</strong>.<br>Please consult a medical professional for further evaluation and guidance.",
        "Українська": "Ваш ймовірний ризик діабету становить <strong>{percentage}%</strong>.<br>Будь ласка, зверніться до лікаря для подальшої оцінки та порад."
    },
    "result_low_title": {
        "English": "✅ Low Diabetes Risk",
        "Українська": "✅ Низький ризик діабету"
    },
    "result_low_text": {
        "English": "Your estimated risk of diabetes is <strong>{percentage}%</strong>.<br>Keep maintaining a healthy lifestyle and monitor your health regularly.",
        "Українська": "Ваш ймовірний ризик діабету становить <strong>{percentage}%</strong>.<br>Продовжуйте вести здоровий спосіб життя і регулярно перевіряйте стан здоров'я."
    },
    "lab_table": {
    "indicator": {
        "English": "Indicator",
        "Українська": "Показник"
    },
    "your_value": {
        "English": "Your Value",
        "Українська": "Ваше значення"
    },
    "normal_range": {
        "English": "Normal Range",
        "Українська": "Норма"
    },
    "status": {
        "English": "Status",
        "Українська": "Статус"
    },
    "statuses": {
        "low": {
            "English": "🔽 Low",
            "Українська": "🔽 Низький"
        },
        "high": {
            "English": "🔼 High",
            "Українська": "🔼 Високий"
        },
        "normal": {
            "English": "✅ Normal",
            "Українська": "✅ Норма"
        }
    },
    "section_title": {
        "English": "📋 Comparison with Medical Norms",
        "Українська": "📋 Порівняння з медичними нормами"
    }
    },
    "labels": {
        "Urea (mmol/L)": {
            "English": "Urea (mmol/L)",
            "Українська": "Сечовина (ммоль/л)"
        },
        "Creatinine (mmol/L)": {
            "English": "Creatinine (mmol/L)",
            "Українська": "Креатинін (ммоль/л)"
        },
        "Glycated Hemoglobin (%)": {
            "English": "Glycated Hemoglobin (%)",
            "Українська": "Глікований гемоглобін (%)"
        },
        "Cholesterol (mmol/L)": {
            "English": "Cholesterol (mmol/L)",
            "Українська": "Холестерин (ммоль/л)"
        },
        "Triglycerides (mmol/L)": {
            "English": "Triglycerides (mmol/L)",
            "Українська": "Тригліцериди (ммоль/л)"
        },
        "High-density Lipoprotein (mmol/L)": {
            "English": "High-density Lipoprotein (mmol/L)",
            "Українська": "ЛПВЩ (ммоль/л)"
        },
        "Low-density Lipoprotein (mmol/L)": {
            "English": "Low-density Lipoprotein (mmol/L)",
            "Українська": "ЛПНЩ (ммоль/л)"
        },
        "Very Low-density Lipoprotein (mmol/L)": {
            "English": "Very Low-density Lipoprotein (mmol/L)",
            "Українська": "ЛДНЩ (ммоль/л)"
        },
        "BMI (kg/m²)": {
            "English": "BMI (kg/m²)",
            "Українська": "ІМТ (кг/м²)"
        }
    }
}



st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        html, body, [class*="st-"] {
            font-family: 'Poppins', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #6f6cd3, #afafee);
            font-family: 'Poppins', sans-serif;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        header, footer {
            visibility: hidden;
        }
        h1 {
            color: white;
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        h3 {
            color: #246bfd;
        }
        .input-card {
            background-color: white;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            margin: auto;
            max-width: 800px;
        }
        }
            
    </style>
""", unsafe_allow_html=True)


with open("logo.png", "rb") as f:
    img_bytes = f.read()
    encoded = base64.b64encode(img_bytes).decode()

st.markdown(
    f"""
    <div style="text-align: center; margin-top: 30px; margin-bottom: 20px;">
        <img src="data:image/png;base64,{encoded}" width="400" />
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("""
<style>
.intro-text {
    color: #ffffff;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
    font-size: 18px;
    line-height: 1.6;
}
.intro-text h3 {
    font-size: 30px;
    font-weight: bold;
    margin-bottom: 10px;
    color: #ffffff;
}
.intro-text ul {
    margin-left: 20px;
    padding-left: 15px;
}
.intro-text li {
    margin-bottom: 10px;
}
.warning {
    color: #ffd700;
    font-weight: bold;
    font-size: 16px;
    margin-top: 20px;
    display: block;
}
</style>
""", unsafe_allow_html=True)
st.markdown(f"""
<div class="intro-text">
    <h3 style="text-align: center;">{T['intro_title'][language]}</h3>
    <p>{T['intro_text'][language]}</p>
    <p><b>{'Our goal is to help you:' if language == 'English' else 'Наша ціль допомогти Вам:'}</b></p>
    <ul>
        {''.join([f'<li>{goal}</li>' for goal in T['goal_list'][language]])}
    </ul>
    <span class="warning">⚠️ {T['disclaimer'][language]}</span>
    <p style="margin-top: 1.5em; font-weight: bold; color: #ffffff;">
        🧪 {T['enter_prompt'][language]}
    </p>
</div>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
        .form-grid {
            display: flex;
            justify-content: center;
            gap: 40px;
            flex-wrap: wrap;
            margin-top: 0.5px;
        }
        .form-column {
            width: 400px;
        }

        .stSelectbox label, .stSlider label, .stNumberInput label {
            color: white !important;
            font-weight: 500;
            font-size: 16px;
        }

        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            font-size: 16px;
            font-weight: 500;
            color: white;
            margin-bottom: 6px;
            display: block;
        }

        .form-group input, .form-group select {
            width: 100%;
            padding: 8px 12px;
            font-size: 16px;
            border-radius: 6px;
        }

    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='form-grid'>", unsafe_allow_html=True)


with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='form-column'>", unsafe_allow_html=True)
 
        gender_options = ["Male", "Female"] if language == "English" else ["Чоловіча", "Жіноча"]

  
        gender_display = st.selectbox(T["form_labels"][language]["gender"], options=gender_options)


        gender = "Male" if gender_display in ["Male", "Чоловіча"] else "Female"

        age = st.slider(T["form_labels"][language]["age"], 1, 120)
        urea = st.number_input(T["form_labels"][language]["urea"], min_value=0.0, step=0.1)
        creatinine = st.number_input(T["form_labels"][language]["creatinine"], min_value=0.0, step=0.1)
        hba1c = st.number_input(T["form_labels"][language]["hba1c"], min_value=0.0, step=0.1)
        bmi = st.number_input(T["form_labels"][language]["bmi"], min_value=0.0, step=0.1)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='form-column'>", unsafe_allow_html=True)
        chol = st.number_input(T["form_labels"][language]["chol"], min_value=0.0, step=0.1)
        tg = st.number_input(T["form_labels"][language]["tg"], min_value=0.0, step=0.1)
        hdl = st.number_input(T["form_labels"][language]["hdl"], min_value=0.0, step=0.1)
        ldl = st.number_input(T["form_labels"][language]["ldl"], min_value=0.0, step=0.1)
        vldl = st.number_input(T["form_labels"][language]["vldl"], min_value=0.0, step=0.1)
        st.markdown("</div>", unsafe_allow_html=True)




gender_encoded = 1 if gender == "Male" else 0
input_data = np.array([[gender_encoded, age, urea, creatinine, hba1c, chol, tg, hdl, ldl, vldl, bmi]])


st.markdown("""
    <style>
      
        .stButton {
            display: flex;
            justify-content: center;
            margin-top: -10px; 
            margin-bottom: 10px;
        }

        .stButton button {
            background-color: #292680;
            color: white;
            padding: 14px 60px;
            font-size: 18px;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }

        .stButton button:hover {
            background-color: #6f6cd3;
        }
    </style>
""", unsafe_allow_html=True)


user_values = {
    "Urea (mmol/L)": urea,
    "Creatinine (mmol/L)": creatinine,
    "Glycated Hemoglobin (%)": hba1c,
    "Cholesterol (mmol/L)": chol,
    "Triglycerides (mmol/L)": tg,
    "High-density Lipoprotein (mmol/L)": hdl,
    "Low-density Lipoprotein (mmol/L)": ldl,
    "Very Low-density Lipoprotein (mmol/L)": vldl,
    "BMI (kg/m²)": bmi,
}

scaler = joblib.load("scaler_pca.pkl")
pca = joblib.load("pca.pkl")

if st.button(T["predict_button"][language]):

    required_non_zero_fields = [
        "Urea (mmol/L)",
        "Creatinine (mmol/L)",
        "Glycated Hemoglobin (%)",
        "High-density Lipoprotein (mmol/L)",
        "BMI (kg/m²)"
    ]

    zero_invalid = any(user_values[key] == 0 for key in required_non_zero_fields)

    if zero_invalid:
        st.warning(T["zero_warning"][language])
    else:
        input_dict = {
            "ID": 0,  
            "No_Pation": 0,  
            "Gender": gender_encoded,
            "AGE": age,
            "Urea": urea,
            "Cr": creatinine,
            "HbA1c": hba1c,
            "Chol": chol,
            "TG": tg,
            "HDL": hdl,
            "LDL": ldl,
            "VLDL": vldl,
            "BMI": bmi
        }

        input_df = pd.DataFrame([input_dict])

    
        input_scaled = scaler.transform(input_df)
        input_pca = pca.transform(input_scaled)

     
        proba = model.predict_proba(input_pca)[0][1]
        percentage = round(proba * 100, 1)

        if proba >= 0.5:
            st.markdown(f"""
                <div style="
                    background-color: #ffe6e6;
                    padding: 25px;
                    border-radius: 15px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    text-align: center;
                    border-left: 6px solid #cc0000;
                    ">
                    <h2 style="color: #b30000; margin-bottom: 10px;">{T['result_high_title'][language]}</h2>
                    <p style="font-size: 17px; color: #4d0000;">
                        {T['result_high_text'][language].format(percentage=percentage)}
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="
                    background-color: #e6f9ec;
                    padding: 25px;
                    border-radius: 15px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    text-align: center;
                    border-left: 6px solid #2e8b57;
                    ">
                    <h2 style="color: #2e8b57; margin-bottom: 10px;">{T['result_low_title'][language]}</h2>
                    <p style="font-size: 17px; color: #225d3d;">
                        {T['result_low_text'][language].format(percentage=percentage)}
                    </p>
                </div>
            """, unsafe_allow_html=True)

        def generate_lab_comparison_table(user_input):
            if gender == "Male":
                normal_ranges = {
                    "Urea (mmol/L)": (2.5, 7.5),
                    "Creatinine (mmol/L)": (64, 104),
                    "Glycated Hemoglobin (%)": (4.0, 5.6),
                    "Cholesterol (mmol/L)": (0.0, 5.17),
                    "Triglycerides (mmol/L)": (0.0, 1.7),
                    "High-density Lipoprotein (mmol/L)": (1.03, 2.3),
                    "Low-density Lipoprotein (mmol/L)": (0.0, 3),
                    "Very Low-density Lipoprotein (mmol/L)": (0.0, 1.0),
                    "BMI (kg/m²)": (18.5, 24.9),
                }
            else:
                normal_ranges = {
                    "Urea (mmol/L)": (2.5, 7.5),
                    "Creatinine (mmol/L)": (43, 90),
                    "Glycated Hemoglobin (%)": (4.0, 5.7),
                    "Cholesterol (mmol/L)": (0.0, 5.17),
                    "Triglycerides (mmol/L)": (0.0, 1.7),
                    "High-density Lipoprotein (mmol/L)": (1.03, 2.3),
                    "Low-density Lipoprotein (mmol/L)": (0.0, 3),
                    "Very Low-density Lipoprotein (mmol/L)": (0.0, 1.0),
                    "BMI (kg/m²)": (18.5, 24.9),
                }

            comparison = []
            for label, (min_val, max_val) in normal_ranges.items():
                value = user_input[label]
                if value < min_val:
                    status = T["lab_table"]["statuses"]["low"][language]
                elif value > max_val:
                    status = T["lab_table"]["statuses"]["high"][language]
                else:
                    status = T["lab_table"]["statuses"]["normal"][language]
                comparison.append({
                    T["lab_table"]["indicator"][language]: T["labels"][label][language],
                    T["lab_table"]["your_value"][language]: value,
                    T["lab_table"]["normal_range"][language]: f"{min_val} - {max_val}",
                    T["lab_table"]["status"][language]: status
                })
            return pd.DataFrame(comparison)


        def render_comparison_html_table(df):
            status_col = T["lab_table"]["status"][language]
            html = """
            <style>
                table.custom-table {
                    width: 100%;
                    border-collapse: collapse;
                    font-family: 'Poppins', sans-serif;
                    border-radius: 12px;
                    overflow: hidden;
                    margin-top: 10px;
                }
                .custom-table th, .custom-table td {
                    border: 1px solid #eee;
                    padding: 12px 14px;
                    text-align: center;
                    font-size: 15px;
                }
                .custom-table th {
                    background-color: #f9fafb;
                    color: #333;
                    font-weight: 600;
                }
                .custom-table tr:nth-child(even) {
                    background-color: #f7f7f7;
                }
                .custom-table tr:nth-child(odd) {
                    background-color: #ffffff;
                }
                .custom-table tr:hover {
                    background-color: #eaf3ff;
                }
                .low {
                    color: #d8000c;
                    font-weight: bold;
                }
                .high {
                    color: #e69500;
                    font-weight: bold;
                }
                .normal {
                    color: #007e33;
                    font-weight: bold;
                }
            </style>
            <table class="custom-table">
                <thead>
                    <tr>
            """
            for col in df.columns:
                html += f"<th>{col}</th>"
            html += "</tr></thead><tbody>"

            for _, row in df.iterrows():
                value = row[status_col]
                if "Low" in value or "Низький" in value:
                    status_class = "low"
                elif "High" in value or "Високий" in value:
                    status_class = "high"
                elif "Normal" in value or "Норма" in value:
                    status_class = "normal"
                else:
                    status_class = ""

                html += "<tr>"
                for col in df.columns:
                    css_class = status_class if col == status_col else ""
                    html += f"<td class='{css_class}'>{row[col]}</td>"
                html += "</tr>"

            html += "</tbody></table>"
            return html


       

        df_comparison = generate_lab_comparison_table(user_values)

        st.markdown(f"""
        <h3 style="color: white; font-weight: 600; margin-top: 1rem;">
            {T["lab_table"]["section_title"][language]}
        </h3>
    """, unsafe_allow_html=True)


        st.markdown(render_comparison_html_table(df_comparison), unsafe_allow_html=True)
