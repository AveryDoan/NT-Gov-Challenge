# GOVHACK_2025
# CiteQueryChatbot  

Resources: https://www.youtube.com/watch?v=LRcjlXL9hPA Text to SQL Agent for Data Visualization
Resources: https://www.youtube.com/watch?v=nXuTJrzkn9Q What Happens When You Combine RAG with Text2SQL?

**Team:** Croc and Chips  
**Event:** GovHack 2025, Northern Territory  
**Challenge:** An Accurate and Trustworthy Chatbot for Data Interactions  
**Team Members:** Max (Ngoc Cuong Hoang), Kingsley (Kar Keat Koh), Felix (Tien Minh Nguyen), Avery (Le Quynh Nhu Doan), Jane (Phuong Tran Tran), Echo (Tianhui Ke), Emma (ThiXuan Thanh Tran)  

---

## Overview  
**CiteQueryChatbot** is an accurate and trustworthy AI assistant designed to empower Northern Territory government workers to interrogate complex datasets conversationally while meeting the high accuracy and auditability standards required for official decision-making.  

Unlike advanced AI models like ChatGPT-5 or Roskets, which risk hallucinations, CiteQueryChatbot prioritizes **precision** by converting plain-English queries into **SQL queries**, executed on a lightweight **SQLite database**. This ensures grounded, scope-limited responses with no external guessing, making it ideal for government use where 90% accuracy is insufficient.  

Our solution addresses the challenge of extracting reliable insights from vast government datasets across HR, finance, and operations. It incorporates **trust scoring, question scaffolding, audit trails, and ethical AI practices** to ensure transparency, privacy, and fairness, with a focus on Northern Territory government needs.  

---

## How It Works  
CiteQueryChatbot uses a **Retrieve-Augment-Generate (RAG) pipeline** with Natural Language Processing (NLP) via **NLTK** to deliver precise, auditable answers.  

1. **Dataset Upload:** Users enter an API key and upload datasets (e.g., APS Employee Census 2024, AusTender Procurement Statistics, Portfolio Budget Statements) into an SQLite database.  
2. **Dataset Summary:** Provides statistical summary (averages, outliers, missing values).  
3. **Question Scaffolding:** NLTK suggests tailored questions if input is vague.

   **Lemmatization:** The project imports WordNetLemmatizer and defines lemmatize_sentence to normalize user inputs and dataset column names (e.g., reducing "means" to "mean" or "employees" to "employee"). This helps in matching synonyms and handling variations in natural language queries.  

   **In Suggest_Query:** User input is lemmatized and compared to column names or template keywords using TF-IDF and cosine similarity (from scikit-learn). For example, if a user asks "average days taken by departments," it lemmatizes to match "avg" synonyms and "department" columns, suggesting an SQL template like SELECT department, AVG(days_taken) FROM table GROUP BY department.  

   **In Suggest_Questions:** Lemmatizes suggested questions (e.g., "What are the average days taken by department?") and ranks them by similarity to the user's input, providing 5 top matches to scaffold vague queries.  

4. **SQL Query Generation:** Converts questions into precise SQL (RAG), executed on the SQLite database.

   **Retrieval (Divide into Smaller Parts):** After uploading a dataset (e.g., CSV to SQLite), the system analyzes columns (numeric, categorical, date) and generates query templates (e.g., "group_by": SELECT {column}, AVG({numeric_column}) FROM table GROUP BY {column}). These templates act as "retrieved" structured knowledge from the dataset's metadata.  

   **Augmentation (Prompt + Context):** User query (e.g., "average days by department") is normalized (lemmatized via NLTK, synonym-mapped) and matched to templates using TF-IDF and cosine similarity. It augments the query with dataset context (columns, unique values) to select/fill the best template (e.g., plugging in "department" and "days_taken").  

   **Generation (LLM-like):** Generates an SQL query from the augmented template, executes it on SQLite, computes a trust score (60% data completeness + 40% relevance), and presents results (table + chart) with assumptions/warnings.


5. **Trust Scoring:** Calculates score (0–100%).  
   - 60% based on data quality (missing values, outliers, consistency).  
   - 40% based on question relevance.  
   - Scores <80% trigger refinement suggestions.  
6. **Result Presentation:** Returns raw data (tables) and graphs (bar, with future pie/line/scatter).  
7. **Audit Trails:** Logs all queries, SQL, trust scores, and metadata for accountability.  

---

## Key Features  

- **Conversational Data Interrogation:** Works across HR, finance, etc.  
- **Trust Scoring:** Reliability ensured via quality (60%) + relevance (40%) + Data validation (integrating) 
- **Grounded Responses:** SQL eliminates hallucinations.  
- **Transferable Framework:** Role-based expansion planned.  
- **Question Scaffolding:** NLP-driven query suggestions.  
- **Audit Trails:** Full transparency with logs.  
- **Ethical AI:** Privacy-preserving, bias-resistant, transparent.  

---

## Dataset Integration  

Supports large-scale government datasets, including:  
- **APS Employee Census 2024** → HR analytics.  
- **AusTender Procurement Statistics** → Operations insights.  
- **Portfolio Budget Statements** → Financial forecasting.  

Accuracy tested on **Kaggle Employee Leave Tracking Dataset** (>1,000 rows), with **100% manual verification match** for queries like:  
> “Which departments show high absenteeism?” (Trust score ≈ 95%).  

---

## Ethical AI Practices  

- **Privacy:** Local processing, no cloud use, complies with Australian Privacy Principles.  
- **Bias Prevention:** Neutral SQL queries, diverse dataset checks.  
- **Transparency:** Trust scores, audit logs, explainable SQL outputs.  

--- 

## Future Plans  

- Expand visualizations (pie, line, scatter).  
- Support PostgreSQL + live APIs.  
- Add role-based permissions.  
- Integrate Microsoft GraphRag.  
- Collect prompts for department-specific needs.  
- Implement dataset access controls.  

---

