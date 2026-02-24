import os
import re
import pandas as pd
from django.shortcuts import render
from django.conf import settings

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline


# =============================
# LOAD CSV 
# =============================
csv_path = os.path.join(settings.BASE_DIR, "data", "hotel.csv")
df = pd.read_csv(csv_path)

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")


# =============================
# CREATE FAISS (ONLY ONCE)
# =============================
documents = []
for _, row in df.iterrows():
    text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
    documents.append(Document(page_content=text))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(documents, embeddings)

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)


# =============================
# HELPER FUNCTIONS
# =============================

def is_structured_question(question):
    keywords = [
        "total", "available", "bookings", "guest",
        "rooms", "countries", "revenue", "from"
    ]
    date_pattern = re.search(r"\d{2}/\d{2}/\d{4}", question)
    return any(word in question for word in keywords) or bool(date_pattern)


def extract_country(question):
    match = re.search(r"from\s+([a-zA-Z\s]+)", question)
    if match:
        return match.group(1).strip().lower()
    return None


def extract_date(question):
    match = re.search(r"\d{2}/\d{2}/\d{4}", question)
    if match:
        return pd.to_datetime(match.group(), dayfirst=True)
    return None


# =============================
# MAIN VIEW
# =============================

def home(request):
    answer = ""

    if request.method == "POST":
        question = request.POST.get("question", "").lower()

        # ---------- Structured Queries ----------
        if is_structured_question(question):

            months_list = [
                "january","february","march","april","may","june",
                "july","august","september","october","november","december"
            ]

            # Month-wise Revenue
            if "revenue" in question and any(month in question for month in months_list):

                months = {m: i+1 for i, m in enumerate(months_list)}

                month_found = None
                for m in months:
                    if m in question:
                        month_found = m
                        break

                if month_found:
                    month_num = months[month_found]

                    year_match = re.search(r"\b(20\d{2})\b", question)
                    if year_match:
                        year = int(year_match.group(1))
                    else:
                        year = pd.Timestamp.now().year

                    mask = (
                        (df["Date"].dt.month == month_num) &
                        (df["Date"].dt.year == year)
                    )

                    month_data = df[mask]

                    if len(month_data) > 0:
                        total_month_revenue = month_data["Revenue"].sum()
                        answer = f"💡 Total revenue of {month_found.capitalize()} {year} is {total_month_revenue}"
                    else:
                        answer = f"❌ No revenue data found for {month_found.capitalize()} {year}"

            # Total Revenue
            elif "total revenue" in question:
                total = df["Revenue"].sum()
                answer = f"💡 Total revenue of dataset is {total}"

            # Rooms Available on a Date
            elif "available" in question and "rooms" in question:
                date = extract_date(question)
                if date is not None:
                    data = df[df["Date"] == date]
                    if len(data) > 0:
                        rooms = data["Available_Rooms"].values[0]
                        answer = f"💡 Available rooms on {date.strftime('%d/%m/%Y')} are {rooms}"
                    else:
                        answer = f"❌ Date {date.strftime('%d/%m/%Y')} not found"
                else:
                    answer = "❌ Please provide date in DD/MM/YYYY format"

            # Guests from any Country
            elif "guest" in question and "from" in question:
                country = extract_country(question)
                if country:
                    mask = df["Guest_Country"].str.lower().str.contains(country)
                    if mask.any():
                        total_guests = df[mask]["Bookings"].sum()
                        answer = f"💡 Total guests from {country.title()} are {total_guests}"
                    else:
                        answer = f"❌ Country '{country}' not found"
                else:
                    answer = "❌ Please specify country"

            # Revenue from any country
            elif "revenue" in question and "from" in question:
                country = extract_country(question)
                if country:
                    mask = df["Guest_Country"].str.lower().str.contains(country)
                    if mask.any():
                        revenue = df[mask]["Revenue"].sum()
                        answer = f"💡 Total revenue from {country.title()} is {revenue}"
                    else:
                        answer = f"❌ Country '{country}' not found"
                else:
                    answer = "❌ Please specify country"

            # List of Countries
            elif "name" in question and "countries" in question:
                countries = df["Guest_Country"].unique()
                answer = f"💡 Involved countries: {', '.join(countries)}"

            # Total Countries
            elif "how many countries" in question or "countries involved" in question:
                countries_count = df["Guest_Country"].nunique()
                answer = f"💡 Total countries involved are {countries_count}"

            # Most bookings
            elif "most bookings" in question:
                top_country = df.groupby("Guest_Country")["Bookings"].sum().idxmax()
                answer = f"💡 Country with most bookings is {top_country}"

            # Total Bookings
            elif "total bookings" in question:
                total_bookings = df["Bookings"].sum()
                answer = f"💡 Total bookings in dataset are {total_bookings}"

            else:
                answer = "❌ Unable to parse structured question"

        # ---------- Unstructured / Semantic Queries ----------
        else:
            docs = db.similarity_search(question, k=10)
            context = "\n".join([d.page_content[:300] for d in docs])

            prompt = f"""
Answer using only the context below.
Do not guess. If answer not in context, say 'Not found'.

Context:
{context}

Question:
{question}
"""

            result = generator(prompt, max_length=200)
            answer = f"💡 {result[0]['generated_text']}"

    return render(request, "home.html", {"answer": answer})