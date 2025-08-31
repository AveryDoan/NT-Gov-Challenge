import pandas as pd
import sqlite3
import re
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from datetime import datetime

# Configure logging for audit trails
logging.basicConfig(filename='chatbot_audit.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class VCAFChatbot:
    def __init__(self):
        """Initialize the chatbot with an in-memory SQLite database."""
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.datasets = {}
        self.vectorizer = TfidfVectorizer()
        self.trust_threshold = 0.9
        self.query_templates = {}
        self.query_history = []
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
        """Load and preprocess a dataset into SQLite."""
        try:
            dataset_name = os.path.basename(file_path).replace('.csv', '').replace('.xlsx', '').replace(' ', '_').lower()
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path, sheet_name=None)
                sheet_name = list(df.keys())[0]  # Use first sheet
                df = df[sheet_name]
            else:
                df = pd.read_csv(file_path)
            
            df = df.dropna(how='all').fillna(0)
            df.columns = [col.replace(' ', '_').lower() for col in df.columns]
            df.to_sql(dataset_name, self.conn, if_exists='replace', index=False)
            self.datasets[dataset_name] = df
            self.query_templates[dataset_name] = self.generate_query_templates(df, dataset_name)
            logging.info("Loaded dataset: %s", dataset_name)
            return dataset_name
        except Exception as e:
            logging.error("Failed to load dataset %s: %s", file_path, str(e))
            return ""

    def generate_query_templates(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, str]:
        """Dynamically generate query templates based on dataset columns and semantics."""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        date_cols = [col for col in df.columns if 'date' in col.lower() or df[col].dtype == 'datetime64[ns]']
        id_cols = [col for col in df.columns if 'id' in col.lower()]

        templates = {
            "summary_stats": f"SELECT {', '.join([f'AVG({col}) as avg_{col}, COUNT({col}) as count_{col}, MIN({col}) as min_{col}, MAX({col}) as max_{col}' for col in numeric_cols])} FROM {dataset_name}",
            "group_by": f"SELECT {{column}}, COUNT(*) as count, AVG({{numeric_column}}) as avg_value, SUM({{numeric_column}}) as total_value FROM {dataset_name} GROUP BY {{column}}",
            "filter_range": f"SELECT * FROM {dataset_name} WHERE {{numeric_column}} BETWEEN ? AND ?",
            "date_filter": f"SELECT * FROM {dataset_name} WHERE {{date_column}} LIKE ?"
        }

        # Add templates for specific datasets
        if 'employee_leave' in dataset_name:
            templates.update({
                "leave_patterns": f"SELECT department, leave_type, AVG(days_taken) as avg_days, COUNT(*) as count, SUM(days_taken) as total_days FROM {dataset_name} WHERE department = ? GROUP BY department, leave_type",
                "leave_balance": f"SELECT employee_name, department, remaining_leaves FROM {dataset_name} WHERE department = ? AND remaining_leaves < ?",
                "leave_by_date": f"SELECT department, leave_type, AVG(days_taken) as avg_days FROM {dataset_name} WHERE start_date LIKE ? GROUP BY department, leave_type"
            })
        if 'tbl_employee' in dataset_name or 'tbl_action' in dataset_name:
            templates.update({
                "terminations": f"SELECT COUNT(*) as termination_count FROM tbl_action WHERE actionid = 30 AND empid IN (SELECT empid FROM tbl_employee WHERE depid = ?) AND effectivedt LIKE ?",
                "performance": f"SELECT AVG(rating) as avg_rating, COUNT(*) as count FROM tbl_perf WHERE empid IN (SELECT empid FROM tbl_employee WHERE depid = ?)",
                "employee_joins": f"SELECT e.empname, e.depid, a.actionid, a.effectivedt FROM tbl_employee e JOIN tbl_action a ON e.empid = a.empid WHERE e.depid = ?"
            })

        # Add join templates if multiple datasets with ID columns are present
        if len(self.datasets) > 1 and id_cols:
            for other_dataset in self.datasets:
                if other_dataset != dataset_name and any(col in self.datasets[other_dataset].columns for col in id_cols):
                    templates[f"join_{other_dataset}"] = f"SELECT * FROM {dataset_name} d1 JOIN {other_dataset} d2 ON d1.{{id_column}} = d2.{{id_column}} WHERE {{condition}}"
        return templates

    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Return column names and basic statistics for a dataset."""
        if dataset_name not in self.datasets:
            return {}
        df = self.datasets[dataset_name]
        return {
            'columns': list(df.columns),
            'stats': df.describe().to_dict(),
            'unique_values': {col: df[col].unique().tolist()[:10] for col in df.select_dtypes(include=['object']).columns}
        }

    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate user query to prevent unsafe SQL operations."""
        unsafe_patterns = [r'\b(DROP|DELETE|UPDATE|INSERT)\b', r'\b(ALTER|TRUNCATE)\b']
        for pattern in unsafe_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                logging.warning("Unsafe query detected: %s", query)
                return False, "Query contains unsafe operations."
        return True, ""

    def compute_trust_score(self, response_data: pd.DataFrame, query: str) -> float:
        """Calculate trust score based on data completeness and query relevance."""
        completeness = response_data.notnull().mean().mean() if not response_data.empty else 0
        query_vector = self.vectorizer.fit_transform([query])
        data_desc = ' '.join(response_data.columns) if not response_data.empty else ''
        data_vector = self.vectorizer.transform([data_desc])
        relevance = cosine_similarity(query_vector, data_vector)[0][0] if data_desc else 0
        trust_score = 0.6 * completeness + 0.4 * relevance
        logging.info("Trust score for query '%s': %.2f", query, trust_score)
        return trust_score

    def execute_query(self, query: str, params: Tuple = ()) -> Tuple[pd.DataFrame, str, float]:
        """Execute a validated SQL query with trust scoring."""
        is_valid, error_msg = self.validate_query(query)
        if not is_valid:
            return pd.DataFrame(), error_msg, 0.0

        try:
            response_data = pd.read_sql_query(query, self.conn, params=params)
            trust_score = self.compute_trust_score(response_data, query)
            if trust_score < self.trust_threshold:
                logging.warning("Low trust score for query '%s': %.2f", query, trust_score)
                return response_data, "Warning: Low confidence in response accuracy.", trust_score
            logging.info("Query executed: %s", query)
            return response_data, "Success", trust_score
        except Exception as e:
            logging.error("Query execution failed: %s, Error: %s", query, str(e))
            return pd.DataFrame(), f"Query failed: {str(e)}", 0.0

    def suggest_query(self, user_input: str, dataset_name: str) -> Tuple[str, Tuple]:
        """Map natural language input to a query template with improved matching."""
        user_input = user_input.lower()
        df = self.datasets.get(dataset_name, pd.DataFrame())
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        date_cols = [col for col in df.columns if 'date' in col.lower() or df[col].dtype == 'datetime64[ns]']
        info = self.get_dataset_info(dataset_name)

        # Normalize user input using synonyms
        for key, synonyms in self.keyword_synonyms.items():
            for synonym in synonyms:
                user_input = user_input.replace(synonym, key)

        # Find best matching template
        template_scores = []
        for template_name, query in self.query_templates.get(dataset_name, {}).items():
            keywords = template_name.split('_') + query.lower().split()
            input_vector = self.vectorizer.fit_transform([user_input])
            template_vector = self.vectorizer.transform([' '.join(keywords)])
            score = cosine_similarity(input_vector, template_vector)[0][0]
            template_scores.append((template_name, query, score))

        # Sort templates by score and pick the best
        template_scores.sort(key=lambda x: x[2], reverse=True)
        if not template_scores or template_scores[0][2] < 0.1:
            query = f"SELECT * FROM {dataset_name} LIMIT 10"
            params = ()
            return query, params

        selected_template, query, score = template_scores[0]
        params = ()

        # Parameter extraction
        if selected_template == "summary_stats":
            params = ()
        elif selected_template == "group_by":
            group_col = categorical_cols[0] if categorical_cols else 'department'
            num_col = numeric_cols[0] if numeric_cols else 'days_taken'
            query = query.replace('{column}', group_col).replace('{numeric_column}', num_col)
            params = ()
        elif selected_template == "filter_range":
            num_col = numeric_cols[0] if numeric_cols else 'days_taken'
            query = query.replace('{numeric_column}', num_col)
            params = (5, 10)  # Example range
        elif selected_template == "date_filter":
            date_col = date_cols[0] if date_cols else 'start_date'
            query = query.replace('{date_column}', date_col)
            year_match = re.search(r'\d{4}', user_input)
            params = (f"%{year_match.group(0)}%" if year_match else "%2023%",)
        elif selected_template == "leave_patterns":
            dept = user_input.split("in ")[-1].split()[0].capitalize() if "in " in user_input else info['unique_values'].get('department', ['Finance'])[0]
            query = query.replace('{column}', 'department').replace('{numeric_column}', 'days_taken')
            params = (dept,)
        elif selected_template == "leave_balance":
            dept = user_input.split("in ")[-1].split()[0].capitalize() if "in " in user_input else info['unique_values'].get('department', ['Finance'])[0]
            threshold = re.search(r'\d+', user_input) or 5
            params = (dept, int(threshold.group(0)) if isinstance(threshold, re.Match) else 5)
        elif selected_template == "terminations":
            params = (self.get_depid(user_input), "%2023%")
        elif selected_template == "performance":
            params = (self.get_depid(user_input),)
        elif selected_template.startswith("join_"):
            id_col = [col for col in df.columns if 'id' in col.lower()][0] if any('id' in col.lower() for col in df.columns) else 'empid'
            query = query.replace('{id_column}', id_col).replace('{condition}', '1=1')
            params = ()

        # Add to query history
        self.query_history.append((user_input, query, params))
        if len(self.query_history) > 10:
            self.query_history.pop(0)

        logging.info("Suggested query for '%s' on dataset %s: %s (score: %.2f)", user_input, dataset_name, query, score)
        return query, params

    def get_depid(self, user_input: str) -> int:
        """Map department name to DepID (simplified mapping)."""
        dept_map = {"Finance": 1, "HR": 2, "Operations": 3, "IT": 4, "Marketing": 5}
        for dept in dept_map:
            if dept.lower() in user_input.lower():
                return dept_map[dept]
        return 1  # Default to Finance

    def close(self):
        """Close database connection."""
        self.conn.close()
        logging.info("Database connection closed.")

class VCAFApp:
    def __init__(self, root):
        """Initialize the Tkinter GUI."""
        self.root = root
        self.root.title("VCAF Data Analytics App")
        self.chatbot = VCAFChatbot()
        self.current_dataset = None
        self.suggested_query = None
        self.suggested_params = None

        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Upload dataset button
        ttk.Button(self.main_frame, text="Upload Dataset", command=self.upload_dataset).grid(row=0, column=0, columnspan=2, pady=5)

        # Dataset info display
        self.dataset_info_text = tk.Text(self.main_frame, height=10, width=80)
        self.dataset_info_text.grid(row=1, column=0, columnspan=2, pady=5)

        # Query input
        ttk.Label(self.main_frame, text="Enter Question:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.query_entry = ttk.Entry(self.main_frame, width=60)
        self.query_entry.grid(row=2, column=1, pady=5)
        ttk.Button(self.main_frame, text="Suggest Query", command=self.suggest_query).grid(row=3, column=0, columnspan=2, pady=5)

        # Suggested query display
        self.query_text = tk.Text(self.main_frame, height=3, width=80)
        self.query_text.grid(row=4, column=0, columnspan=2, pady=5)
        ttk.Button(self.main_frame, text="Run Query", command=self.run_query).grid(row=5, column=0, columnspan=2, pady=5)

        # Results display
        self.result_text = tk.Text(self.main_frame, height=10, width=80)
        self.result_text.grid(row=6, column=0, columnspan=2, pady=5)

    def upload_dataset(self):
        """Handle dataset upload."""
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
        """Display column names and statistics for the current dataset."""
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

    def suggest_query(self):
        """Suggest an SQL query based on user input."""
        if not self.current_dataset:
            messagebox.showerror("Error", "No dataset loaded. Please upload a dataset.")
            return
        user_input = self.query_entry.get()
        if not user_input:
            messagebox.showerror("Error", "Please enter a question.")
            return
        self.suggested_query, self.suggested_params = self.chatbot.suggest_query(user_input, self.current_dataset)
        self.query_text.delete(1.0, tk.END)
        self.query_text.insert(tk.END, f"Suggested Query: {self.suggested_query}\nParameters: {self.suggested_params}")

    def run_query(self):
        """Execute the suggested query and display results."""
        if not self.suggested_query:
            messagebox.showerror("Error", "No query suggested. Please suggest a query first.")
            return
        result, message, trust_score = self.chatbot.execute_query(self.suggested_query, self.suggested_params)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Message: {message}\n")
        self.result_text.insert(tk.END, f"Trust Score: {trust_score:.2f}\n")
        if not result.empty:
            self.result_text.insert(tk.END, "Results:\n")
            self.result_text.insert(tk.END, result.to_string(index=False))
        else:
            self.result_text.insert(tk.END, "No data returned or query failed.")

if __name__ == "__main__":
    root = tk.Tk()
    app = VCAFApp(root)
    root.mainloop()