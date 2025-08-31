import pandas as pd
import sqlite3
import os
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
from sqlalchemy import create_engine, inspect
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
import json
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(filename='chatbot_audit.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Adapted from State.py
class InputState(TypedDict):
    question: str
    uuid: str
    parsed_question: Dict[str, Any]
    unique_nouns: List[str]
    sql_query: str

class OutputState(TypedDict):
    parsed_question: Dict[str, Any]
    unique_nouns: List[str]
    sql_query: str
    sql_valid: bool
    sql_issues: str

# Adapted from DatabaseManager.py
class DatabaseManager:
    def __init__(self, conn: sqlite3.Connection, engine):
        self.conn = conn
        self.engine = engine

    def get_schema(self, uuid: str) -> str:
        """Retrieve the database schema for the given table."""
        try:
            inspector = inspect(self.engine)
            schema = ""
            table_name = uuid  # Treat uuid as table name
            if table_name in inspector.get_table_names():
                columns = inspector.get_columns(table_name)
                schema += f"Table '{table_name}':\n"
                schema += "Columns:\n" + "\n".join([f"  - {col['name']}: {col['type']}" for col in columns])
            return schema
        except Exception as e:
            raise Exception(f"Error fetching schema: {str(e)}")

    def execute_query(self, uuid: str, query: str) -> List[Any]:
        """Execute SQL query on the database and return results."""
        try:
            with self.conn:
                cursor = self.conn.cursor()
                cursor.execute(query)
                results = cursor.fetchall()
                return results
        except Exception as e:
            raise Exception(f"Error executing query: {str(e)}")

# Adapted from LLMManager.py
class LLMManager:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key="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"
        )

    def invoke(self, prompt: ChatPromptTemplate, **kwargs) -> str:
        messages = prompt.format_messages(**kwargs)
        response = self.llm.invoke(messages)
        return response.content

# Adapted from SQLAgent.py
class SQLAgent:
    def __init__(self, db_manager: DatabaseManager, llm_manager: LLMManager):
        self.db_manager = db_manager
        self.llm_manager = llm_manager

    def parse_question(self, state: dict) -> dict:
        question = state['question']
        schema = self.db_manager.get_schema(state['uuid'])
        prompt = ChatPromptTemplate.from_messages([
            ("system", '''
You are a data analyst. Given a question and database schema, identify relevant tables and columns.
Respond in JSON: {"relevant_tables": [{"table_name": str, "columns": [str], "noun_columns": [str]}]}
'''),
            ("human", "Schema:\n{schema}\nQuestion:\n{question}\nIdentify relevant tables and columns:")
        ])
        response = self.llm_manager.invoke(prompt, schema=schema, question=question)
        return {"parsed_question": json.loads(response)}

    def get_unique_nouns(self, state: dict) -> dict:
        parsed_question = state['parsed_question']
        unique_nouns = set()
        for table_info in parsed_question['relevant_tables']:
            table_name = table_info['table_name']
            noun_columns = table_info['noun_columns']
            if noun_columns:
                columns = ', '.join(f"`{col}`" for col in noun_columns)
                query = f"SELECT DISTINCT {columns} FROM `{table_name}`"
                results = self.db_manager.execute_query(state['uuid'], query)
                for row in results:
                    unique_nouns.update(str(value) for value in row if value)
        return {"unique_nouns": list(unique_nouns)}

    def generate_sql(self, state: dict) -> dict:
        question = state['question']
        parsed_question = state['parsed_question']
        unique_nouns = state['unique_nouns']
        schema = self.db_manager.get_schema(state['uuid'])
        prompt = ChatPromptTemplate.from_messages([
            ("system", '''
Generate a valid SQL query for the question using the schema and unique nouns.
Return a query string.
Skip rows where any column is NULL, "N/A", or "".
Example: SELECT `product_name`, SUM(`quantity`) AS total_quantity FROM `sales` WHERE `product_name` IS NOT NULL AND `quantity` IS NOT NULL GROUP BY `product_name`
'''),
            ("human", "Schema:\n{schema}\nQuestion:\n{question}\nRelevant tables:\n{parsed_question}\nNouns:\n{unique_nouns}\nGenerate SQL query string:")
        ])
        response = self.llm_manager.invoke(prompt, schema=schema, question=question, parsed_question=parsed_question, unique_nouns=unique_nouns)
        return {"sql_query": response if response.strip() != "NOT_ENOUGH_INFO" else "NOT_RELEVANT"}

    def validate_and_fix_sql(self, state: dict) -> dict:
        sql_query = state['sql_query']
        if sql_query == "NOT_RELEVANT":
            return {"sql_query": "NOT_RELEVANT", "sql_valid": False}
        schema = self.db_manager.get_schema(state['uuid'])
        prompt = ChatPromptTemplate.from_messages([
            ("system", '''
Validate and fix the SQL query. Ensure table/column names are in backticks.
Respond in JSON: {"valid": bool, "issues": str or null, "corrected_query": str}
'''),
            ("human", "Schema:\n{schema}\nQuery:\n{sql_query}\nRespond in JSON:")
        ])
        response = self.llm_manager.invoke(prompt, schema=schema, sql_query=sql_query)
        result = json.loads(response)
        return {
            "sql_query": result["corrected_query"] if not result["valid"] else sql_query,
            "sql_valid": result["valid"],
            "sql_issues": result["issues"]
        }

# Adapted from WorkflowManager.py
class WorkflowManager:
    def __init__(self, conn, engine):
        self.sql_agent = SQLAgent(DatabaseManager(conn, engine), LLMManager())

    def create_workflow(self) -> StateGraph:
        # Updated to use state_schema instead of input/output
        workflow = StateGraph(state_schema=InputState)
        workflow.add_node("parse_question", self.sql_agent.parse_question)
        workflow.add_node("get_unique_nouns", self.sql_agent.get_unique_nouns)
        workflow.add_node("generate_sql", self.sql_agent.generate_sql)
        workflow.add_node("validate_and_fix_sql", self.sql_agent.validate_and_fix_sql)
        workflow.add_edge("parse_question", "get_unique_nouns")
        workflow.add_edge("get_unique_nouns", "generate_sql")
        workflow.add_edge("generate_sql", "validate_and_fix_sql")
        workflow.add_edge("validate_and_fix_sql", END)
        workflow.set_entry_point("parse_question")
        return workflow

    def run_sql_agent(self, question: str, uuid: str) -> dict:
        app = self.create_workflow().compile()
        result = app.invoke({"question": question, "uuid": uuid})
        return {
            "sql_query": result['sql_query'],
            "sql_valid": result['sql_valid'],
            "sql_issues": result['sql_issues']
        }

# Main VCAFChatbot class
class VCAFChatbot:
    def __init__(self):
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.engine = create_engine('sqlite:///:memory:')
        self.datasets = {}
        self.workflow_manager = WorkflowManager(self.conn, self.engine)
        self.vectorizer = TfidfVectorizer()
        self.suggested_questions = [
            "What's the average performance rating?",
            "Show low performing employees.",
            "List terminations by year.",
            "What's happening with leave patterns in Finance?",
            "Show employees with low remaining leaves.",
            "Analyze leave by department.",
            "Show outliers in days taken.",
            "Group performance by department.",
            "Filter leaves in 2023.",
            "Join employee and action data."
        ]
        self.keyword_synonyms = {
            'show': ['display', 'list', 'view'],
            'average': ['mean', 'avg'],
            'total': ['sum', 'count'],
            'group': ['by', 'categorized'],
            'low': ['below', 'less than'],
            'high': ['above', 'greater than'],
            'leave': ['absence', 'time off'],
            'performance': ['rating', 'evaluation']
        }
        logging.info("VCAFChatbot initialized")

    def load_dataset(self, file_path: str) -> str:
        try:
            dataset_name = os.path.basename(file_path).replace('.csv', '').replace('.xlsx', '').replace(' ', '_').lower()
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path, sheet_name=None)
                sheet_name = list(df.keys())[0]
                df = df[sheet_name]
            else:
                df = pd.read_csv(file_path)
            df = df.dropna(how='all').fillna(0)
            df.columns = [col.replace(' ', '_').lower() for col in df.columns]
            df.to_sql(dataset_name, self.conn, if_exists='replace', index=False)
            df.to_sql(dataset_name, self.engine, if_exists='replace', index=False)
            self.datasets[dataset_name] = df
            logging.info("Loaded dataset: %s", dataset_name)
            return dataset_name
        except Exception as e:
            logging.error("Failed to load dataset %s: %s", file_path, str(e))
            return ""

    def get_dataset_info(self, dataset_name: str) -> Dict:
        if dataset_name not in self.datasets:
            return {}
        df = self.datasets[dataset_name]
        return {
            'columns': list(df.columns),
            'stats': df.describe().to_dict(),
            'unique_values': {col: df[col].unique().tolist()[:10] for col in df.select_dtypes(include=['object']).columns}
        }

    def execute_agent(self, question: str, dataset_name: str) -> Tuple[str, bool, str]:
        if not dataset_name:
            return "No dataset loaded.", False, ""
        try:
            result = self.workflow_manager.run_sql_agent(question, dataset_name)
            return result['sql_query'], result['sql_valid'], result['sql_issues'] or ""
        except Exception as e:
            logging.error("Agent execution failed: %s", str(e))
            return f"Failed to generate SQL: {str(e)}", False, str(e)

    def suggest_questions(self, user_input: str) -> List[str]:
        user_input = user_input.lower()
        for key, synonyms in self.keyword_synonyms.items():
            for synonym in synonyms:
                user_input = user_input.replace(synonym, key)
        input_vector = self.vectorizer.fit_transform([user_input])
        q_vectors = self.vectorizer.transform(self.suggested_questions)
        similarities = cosine_similarity(input_vector, q_vectors)[0]
        top_indices = np.argsort(similarities)[-3:][::-1]
        suggestions = [self.suggested_questions[i] for i in top_indices if similarities[i] > 0.1]
        logging.info("Suggested questions for '%s': %s", user_input, suggestions)
        return suggestions

# GUI Application
class VCAFApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VCAF Data Analytics App")
        self.chatbot = VCAFChatbot()
        self.current_dataset = None
        self.selected_question = None
        self.is_custom = True

        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Button(self.main_frame, text="Upload Dataset", command=self.upload_dataset).grid(row=0, column=0, columnspan=2, pady=5)

        self.dataset_info_text = tk.Text(self.main_frame, height=10, width=80)
        self.dataset_info_text.grid(row=1, column=0, columnspan=2, pady=5)

        ttk.Label(self.main_frame, text="Enter Your Question:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.query_entry = ttk.Entry(self.main_frame, width=60)
        self.query_entry.grid(row=2, column=1, pady=5)
        ttk.Button(self.main_frame, text="Get Suggestions", command=self.get_suggestions).grid(row=3, column=0, columnspan=2, pady=5)

        ttk.Label(self.main_frame, text="Suggested Questions:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.suggestions_list = tk.Listbox(self.main_frame, height=5, width=80)
        self.suggestions_list.grid(row=4, column=1, pady=5)
        self.suggestions_list.bind("<<ListboxSelect>>", self.select_suggestion)

        ttk.Button(self.main_frame, text="Process Question", command=self.process_question).grid(row=5, column=0, columnspan=2, pady=5)

        self.result_text = tk.Text(self.main_frame, height=10, width=80)
        self.result_text.grid(row=6, column=0, columnspan=2, pady=5)

        self.sql_text = tk.Text(self.main_frame, height=5, width=80)
        self.sql_text.grid(row=7, column=0, columnspan=2, pady=5)

    def upload_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV/Excel Files", "*.csv *.xlsx")])
        if file_path:
            dataset_name = self.chatbot.load_dataset(file_path)
            if dataset_name:
                self.current_dataset = dataset_name
                self.display_dataset_info()
                messagebox.showinfo("Success", f"Dataset {dataset_name} loaded successfully!")
            else:
                messagebox.showerror("Error", "Failed to load dataset.")

    def display_dataset_info(self):
        if not self.current_dataset:
            return
        info = self.chatbot.get_dataset_info(self.current_dataset)
        self.dataset_info_text.delete(1.0, tk.END)
        self.dataset_info_text.insert(tk.END, f"Dataset: {self.current_dataset}\n")
        self.dataset_info_text.insert(tk.END, f"Columns: {', '.join(info['columns'])}\n")
        self.dataset_info_text.insert(tk.END, "Statistics:\n")
        for col, stats in info['stats'].items():
            self.dataset_info_text.insert(tk.END, f"  {col}: Mean = {stats.get('mean', 0):.2f}, Count = {stats.get('count', 0)}\n")
        self.dataset_info_text.insert(tk.END, "Unique Values (sample):\n")
        for col, values in info.get('unique_values', {}).items():
            self.dataset_info_text.insert(tk.END, f"  {col}: {', '.join(map(str, values))}\n")

    def get_suggestions(self):
        if not self.current_dataset:
            messagebox.showerror("Error", "No dataset loaded.")
            return
        user_input = self.query_entry.get()
        if not user_input:
            messagebox.showerror("Error", "Please enter a question.")
            return
        suggestions = self.chatbot.suggest_questions(user_input)
        self.suggestions_list.delete(0, tk.END)
        for sug in suggestions:
            self.suggestions_list.insert(tk.END, sug)
        if suggestions:
            self.suggestions_list.select_set(0)
            self.select_suggestion(None)

    def select_suggestion(self, event):
        selection = self.suggestions_list.curselection()
        if selection:
            self.selected_question = self.suggestions_list.get(selection[0])
            self.is_custom = False
        else:
            self.selected_question = None
            self.is_custom = True

    def process_question(self):
        if not self.current_dataset:
            messagebox.showerror("Error", "No dataset loaded.")
            return
        question = self.selected_question if self.selected_question else self.query_entry.get()
        if not question:
            messagebox.showerror("Error", "No question selected or entered.")
            return

        progress_window = tk.Toplevel(self.root)
        progress_window.title("Processing")
        ttk.Label(progress_window, text="Processing query...").pack(pady=10)
        progress = ttk.Progressbar(progress_window, mode='indeterminate', length=300)
        progress.pack(pady=10)
        progress.start()

        def finish_processing():
            progress.stop()
            progress_window.destroy()
            sql_query, sql_valid, sql_issues = self.chatbot.execute_agent(question, self.current_dataset)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Processed Question: {question}\n")
            if sql_valid:
                self.result_text.insert(tk.END, "SQL generated successfully.\n")
            else:
                self.result_text.insert(tk.END, f"SQL issues: {sql_issues}\n")
                self.result_text.insert(tk.END, "Corrected SQL provided.\n")

            self.sql_text.delete(1.0, tk.END)
            self.sql_text.insert(tk.END, "Generated SQL Query:\n")
            self.sql_text.insert(tk.END, sql_query + "\n")

        self.root.after(1000, finish_processing)  # Reduced time for simulation

if __name__ == "__main__":
    root = tk.Tk()
    app = VCAFApp(root)
    root.mainloop()