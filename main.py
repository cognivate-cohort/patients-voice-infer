import os
import pandas as pd
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import requests
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not set in environment")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)

if os.path.exists("Patient_Display.csv"):
    df = pd.read_csv("Patient_Display.csv")
    token = len(df) + 1
    info = {
        "Names": df["Names"].tolist(),
        "Age": [],
        "Gender": [],
        "Symptoms": [],
        "token no.": [],
        "Doctor": []
    }
else:
    info = {
        "Names": [],
        "Age": [],
        "Gender": [],
        "Symptoms": [],
        "token no.": [],
        "Doctor": []
    }
    token = 100000

def query(input_text):
    prompt = f"""
You are an information extraction engine.
From the following patient message, extract:
Name, Surname, Age, Gender, Symptoms

Message:
"""{input_text}"""

Return output in exactly this one-line format:
NAME SURNAME AGE GENDER SYMPTOM1 SYMPTOM2 ...

Rules:
- ONE line only — no line breaks.
- Use CAPITAL LETTERS only.
- Use only single spaces to separate items.
- Do NOT return anything else. No labels, no punctuation.
- Only return the final line.
"""
    response = llm([HumanMessage(content=prompt)])
    return response.content.strip()

def assignment(symptoms):
    prompt = f"""
out of the given doctor that i am listing choose the appropriate doctor
"""{symptoms}"""
Dr. ABC (Cardiologist)
Dr. DEF (Neurologist)
Dr. GHI (Pulmonologist)
Dr. JKL (Gastroenterologist)
Dr. MNO (Orthopedists)

your result should be as follows 
Dr. ___ (specialty)
"""
    response = llm([HumanMessage(content=prompt)])
    return response.content.strip()

while True:
    duration = 20
    filename = "patient_voice.wav"

    print("Please speak now...")
    audio = sd.rec(frames=duration * 44100, samplerate=44100, channels=1, dtype='int16')
    sd.wait()
    write(filename, 44100, audio)

    print("Please wait while we are processing...")

    model = whisper.load_model("base")
    result = model.transcribe(filename)
    s = result["text"]

    details = query(s)
    parts = details.split(" ", 4)

    name = parts[0] + " " + parts[1]
    age = parts[2]
    gender = parts[3]
    symptoms = parts[4]

    print("Name:", name)
    print("Age:", age)
    print("Gender:", gender)
    print("Symptoms:", symptoms)

    status = input("Are these information correct (press y for yes and n for no): ")
    if status.lower() == 'y':
        assigned = assignment(symptoms)
        print("Your doctor is:")
        print(assigned)
        print(f"Your token no.: P{token}")
        info["Names"].append(name)
        info["Age"].append(age)
        info["Gender"].append(gender)
        info["Symptoms"].append(symptoms)
        info["token no."].append("P" + str(token))
        info["Doctor"].append(assigned)
        token += 1
        break
    else:
        print("Sorry for the inconvenience! Please tell your details again.")

df = pd.DataFrame(info)

df[df["Doctor"] == "Dr. ABC (Cardiologist)"].to_csv("Cardiology(ABC).csv", index=False, encoding='utf-8-sig")
df[df["Doctor"] == "Dr. DEF (Neurologist)"].to_csv("Neurology(DEF).csv", index=False, encoding='utf-8-sig")
df[df["Doctor"] == "Dr. GHI (Pulmonologist)"].to_csv("Pulmonology(GHI).csv", index=False, encoding='utf-8-sig")
df[df["Doctor"] == "Dr. JKL (Gastroenterologist)"].to_csv("Gastroenterology(JKL).csv", index=False, encoding='utf-8-sig")
df[df["Doctor"] == "Dr. MNO (Orthopedists)"].to_csv("Orthopedic(MNO).csv", index=False, encoding='utf-8-sig")

df[["Names", "token no.", "Doctor"]].to_csv("Patient_Display.csv", index=False, encoding='utf-8-sig")
