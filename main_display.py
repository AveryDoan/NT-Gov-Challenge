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
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize_sentence(sentence):
    return ' '.join([lemmatizer.lemmatize(word) for word in sentence.lower().split()])

# Configure logging for audit trails
logging.basicConfig(filename='chatbot_audit.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class CiteQueryChatbot:
    def __init__(self):
        """Initialize the chatbot with an in-memory SQLite database."""
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.datasets = {}
        self.vectorizer = TfidfVectorizer()
        self.trust_threshold = 0.9
        self.query_templates = {}
        self.query_history = []
        self.keyword_synonyms = {
            'show': ['display', 'list', 'view', 'present', 'give', 'find'],
            'average': ['mean', 'avg', 'median'],
            'total': ['sum', 'count', 'number', 'amount'],
            'group': ['by', 'categorized', 'split', 'divide'],
            'low': ['below', 'less than', 'under', 'few'],
            'high': ['above', 'greater than', 'over', 'many'],
            'leave': ['absence', 'time off', 'holiday', 'vacation'],
            'performance': ['rating', 'evaluation', 'score', 'review'],
            'employee': ['staff', 'worker', 'personnel'],
            'termination': ['fired', 'let go', 'dismissed', 'resigned'],
            'department': ['team', 'division', 'section'],
        }
        logging.info("CiteQueryChatbot initialized")

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
            "date_filter": f"SELECT * FROM {dataset_name} WHERE {{date_column}} LIKE ?",
            "sum_filter": f"SELECT SUM({{numeric_column}}) FROM {dataset_name} WHERE {{categorical_column}} = ?",
            "avg_filter": f"SELECT AVG({{numeric_column}}) FROM {dataset_name} WHERE {{categorical_column}} = ?",
            "count_filter": f"SELECT COUNT(*) FROM {dataset_name} WHERE {{categorical_column}} = ?",
            "avg_department": f"SELECT department, AVG(days_taken) as avg_days FROM {dataset_name} GROUP BY department",
        }

        if 'employee_leave' in dataset_name or 'leave' in dataset_name:
            templates.update({
                "leave_balance": f"SELECT employee_name, department, remaining_leaves FROM {dataset_name} WHERE department = ? ORDER BY remaining_leaves ASC",
                "leave_by_month": f"SELECT month, SUM(days_taken) as total_days FROM {dataset_name} GROUP BY month",
                "sensitive_info": f"SELECT * FROM {dataset_name} WHERE department = ? AND (position LIKE '%Manager%' OR position LIKE '%HR%')"
            })
        if 'tbl_employee' in dataset_name or 'tbl_action' in dataset_name:
            templates.update({
                "terminations": f"SELECT COUNT(*) as termination_count FROM tbl_action WHERE actionid = 30 AND empid IN (SELECT empid FROM tbl_employee WHERE depid = ?) AND effectivedt LIKE ?",
                "performance": f"SELECT AVG(rating) as avg_rating, COUNT(*) as count FROM tbl_perf WHERE empid IN (SELECT empid FROM tbl_employee WHERE depid = ?)",
                "employee_joins": f"SELECT e.empname, e.depid, a.actionid, a.effectivedt FROM tbl_employee e JOIN tbl_action a ON e.empid = a.empid WHERE e.depid = ?"
            })

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
                return response_data, "Warning: Low confidence in response accuracy. Please verify with authorized personnel.", trust_score
            logging.info("Query executed: %s", query)
            return response_data, "Success", trust_score
        except Exception as e:
            logging.error("Query execution failed: %s, Error: %s", query, str(e))
            return pd.DataFrame(), f"Query failed: {str(e)}", 0.0

    def suggest_query(self, user_input: str, dataset_name: str) -> Tuple[str, Tuple, str]:
        """Map natural language input to a query template with improved matching."""
        user_input_lower = user_input.lower()
        df = self.datasets.get(dataset_name, pd.DataFrame())
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        date_cols = [col for col in df.columns if 'date' in col.lower() or df[col].dtype == 'datetime64[ns]']
        info = self.get_dataset_info(dataset_name)

        # Normalize user input using synonyms
        for key, synonyms in self.keyword_synonyms.items():
            for synonym in synonyms:
                user_input_lower = user_input_lower.replace(synonym, key)

        # Detect operation
        operation = None
        if 'how many' in user_input_lower or 'total' in user_input_lower or 'sum' in user_input_lower:
            operation = 'SUM'
        elif 'average' in user_input_lower or 'mean' in user_input_lower:
            operation = 'AVG'
        elif 'count' in user_input_lower:
            operation = 'COUNT'

        filter_col = None
        filter_val = None
        if 'department' in user_input_lower:
            filter_col = 'department'
            # Extract department value
            dept_match = re.search(r'in (\w+) department', user_input_lower)
            if dept_match:
                filter_val = dept_match.group(1).capitalize()
            else:
                words = user_input_lower.split()
                for i, word in enumerate(words):
                    if word == 'in' and i+1 < len(words) and words[i+1].isalpha():
                        filter_val = words[i+1].capitalize()
                        break

        # Extract target column phrase
        target_col_part = re.sub(r'in \w+ department', '', user_input_lower).replace('how many', '').replace('total', '').replace('sum', '').replace('average', '').replace('mean', '').replace('count', '').strip()
        target_col_lem = lemmatize_sentence(target_col_part)

        # Match to numeric columns for operation
        if operation and numeric_cols:
            col_lem = [lemmatize_sentence(col) for col in numeric_cols]
            if col_lem:
                self.vectorizer.fit(col_lem + [target_col_lem])
                input_vec = self.vectorizer.transform([target_col_lem])
                col_vec = self.vectorizer.transform(col_lem)
                scores = cosine_similarity(input_vec, col_vec)[0]
                max_score = max(scores)
                if max_score > 0.3:
                    target_col = numeric_cols[np.argmax(scores)]
                    assumption = f"Assuming you mean column '{target_col}' for '{target_col_part}'."
                    if filter_col and filter_val:
                        assumption += f" and department '{filter_val}'."
                        template_key = f"{operation.lower()}_filter".replace('sum', 'sum').replace('avg', 'avg').replace('count', 'count')
                        query = self.query_templates.get(dataset_name, {}).get(template_key, "").replace('{numeric_column}', target_col).replace('{categorical_column}', filter_col)
                        params = (filter_val,)
                    else:
                        # Use avg_by_department for specific "average days by department" query
                        if 'average' in user_input_lower and 'department' in user_input_lower and 'days' in target_col_lem:
                            query = self.query_templates.get(dataset_name, {}).get("avg_by_department", "")
                            params = ()
                            assumption = "Calculating average days taken per department."
                        else:
                            query = f"SELECT {operation}({target_col}) FROM {dataset_name}"
                            params = ()
                    # Add ethical note for sensitive data
                    if 'leave_taken_so_far' in target_col.lower() or 'remaining_leaves' in target_col.lower():
                        assumption += " This information is sensitive; ensure you have proper authorization as per NT government guidelines."
                    logging.info(assumption)
                    return query, params, assumption

        # Fallback to template matching
        template_scores = []
        for template_name, query in self.query_templates.get(dataset_name, {}).items():
            keywords = template_name.split('_') + query.lower().split()
            input_vector = self.vectorizer.fit_transform([user_input_lower])
            template_vector = self.vectorizer.transform([' '.join(keywords)])
            score = cosine_similarity(input_vector, template_vector)[0][0]
            template_scores.append((template_name, query, score))

        template_scores.sort(key=lambda x: x[2], reverse=True)
        if not template_scores or template_scores[0][2] < 0.1:
            query = f"SELECT * FROM {dataset_name} LIMIT 10"
            params = ()
            return query, params, "Showing sample data as fallback."

        selected_template, query, score = template_scores[0]
        params = ()
        assumption = ""

        if selected_template == "summary_stats":
            params = ()
        elif selected_template == "group_by":
            group_col = 'department' if 'department' in user_input_lower else (categorical_cols[0] if categorical_cols else 'department')
            num_col = 'days_taken' if 'days' in user_input_lower else (numeric_cols[0] if numeric_cols else 'days_taken')
            query = query.replace('{column}', group_col).replace('{numeric_column}', num_col)
            params = ()
            assumption = f"Grouping by {group_col} and aggregating {num_col}."
        elif selected_template == "filter_range":
            num_col = numeric_cols[0] if numeric_cols else 'days_taken'
            query = query.replace('{numeric_column}', num_col)
            params = (5, 10)  # Default range; can parse from input
        elif selected_template == "date_filter":
            date_col = date_cols[0] if date_cols else 'start_date'
            query = query.replace('{date_column}', date_col)
            year_match = re.search(r'\d{4}', user_input_lower)
            params = (f"%{year_match.group(0)}%" if year_match else "%2023%",)
        elif selected_template == "leave_patterns":
            params = ()  # No parameters needed for leave_patterns
            assumption = "Analyzing leave patterns across all departments and leave types."
        elif selected_template == "leave_balance":
            dept = filter_val if filter_val else info['unique_values'].get('department', ['Finance'])[0]
            params = (dept,)
            assumption = f"Showing low leave balances for {dept} department. Sensitive data - authorize access."
        elif selected_template == "terminations":
            params = (self.get_depid(user_input_lower), "%2023%")
        elif selected_template == "performance":
            params = (self.get_depid(user_input_lower),)
        elif selected_template.startswith("join_"):
            id_col = [col for col in df.columns if 'id' in col.lower()][0] if any('id' in col.lower() for col in df.columns) else 'empid'
            query = query.replace('{id_column}', id_col).replace('{condition}', '1=1')
            params = ()

        self.query_history.append((user_input, query, params))
        if len(self.query_history) > 10:
            self.query_history.pop(0)

        logging.info("Suggested query for '%s' on dataset %s: %s (score: %.2f)", user_input, dataset_name, query, score)
        return query, params, assumption

    def get_depid(self, user_input: str) -> int:
        """Map department name to DepID (simplified mapping)."""
        dept_map = {"Finance": 1, "HR": 2, "Operations": 3, "IT": 4, "Marketing": 5}
        for dept in dept_map:
            if dept.lower() in user_input.lower():
                return dept_map[dept]
        return 1

    def close(self):
        """Close database connection."""
        self.conn.close()
        logging.info("Database connection closed.")

class CiteQueryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CiteQuery Dashboard - NT Government AI Solution")
        self.chatbot = CiteQueryChatbot()
        self.current_dataset = None
        self.suggested_query = None
        self.suggested_params = None
        self.datasets_list = []

        # Modern color palette
        sidebar_bg = "#1A73E8"  # Google blue
        main_bg = "#F8FAFC"     # Very light gray
        accent = "#F9AB00"      # Accent yellow
        text_color = "#222222"

        # Apply ttk theme and custom styles with larger font sizes
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Sidebar.TFrame', background=sidebar_bg)
        style.configure('Main.TFrame', background=main_bg)
        style.configure('TButton', font=('Segoe UI', 13, 'bold'), padding=8, background=sidebar_bg, foreground='white', borderwidth=0)
        style.map('TButton', background=[('active', accent)], foreground=[('active', text_color)])
        style.configure('Header.TLabel', font=('Segoe UI', 20, 'bold'), background=main_bg, foreground=sidebar_bg)
        style.configure('SidebarHeader.TLabel', font=('Segoe UI', 18, 'bold'), background=sidebar_bg, foreground='white')
        style.configure('TLabel', font=('Segoe UI', 13), background=main_bg, foreground=text_color)
        style.configure('TEntry', font=('Segoe UI', 13), padding=4)
        style.configure('TCombobox', font=('Segoe UI', 13))

        # Main container
        self.container = ttk.Frame(self.root)
        self.container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ttk.Frame(self.container, style='Sidebar.TFrame', width=200)
        self.sidebar.grid(row=0, column=0, sticky=(tk.N, tk.S))
        self.sidebar.grid_propagate(False)
        self.container.columnconfigure(0, weight=0)
        self.container.columnconfigure(1, weight=1)

        # Sidebar header text (no logo)
        ttk.Label(self.sidebar, text="CiteQuery", style='SidebarHeader.TLabel').grid(row=0, column=0, pady=20, padx=20)

        # Sidebar buttons and labels
        ttk.Button(self.sidebar, text="Upload Dataset", command=self.upload_dataset).grid(row=1, column=0, padx=20, pady=5, sticky=tk.EW)
        ttk.Label(self.sidebar, text="Select Dataset:", background=sidebar_bg, foreground='white', font=('Segoe UI', 13)).grid(row=2, column=0, pady=(10, 0), padx=20, sticky=tk.W)
        self.dataset_combo = ttk.Combobox(self.sidebar, values=self.datasets_list, state="readonly", font=('Segoe UI', 13))
        self.dataset_combo.grid(row=3, column=0, padx=20, pady=5, sticky=tk.EW)
        self.dataset_combo.bind("<<ComboboxSelected>>", self.select_dataset)
        ttk.Button(self.sidebar, text="Suggest Query", command=self.suggest_query).grid(row=4, column=0, padx=20, pady=5, sticky=tk.EW)
        ttk.Button(self.sidebar, text="Execute Query", command=self.run_query).grid(row=5, column=0, padx=20, pady=5, sticky=tk.EW)
        ttk.Button(self.sidebar, text="View Audit Log", command=self.view_audit_log).grid(row=6, column=0, padx=20, pady=5, sticky=tk.EW)

        # Main content area
        self.main_frame = ttk.Frame(self.container, style='Main.TFrame', padding=20)
        self.main_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        self.main_frame.rowconfigure(4, weight=1)
        self.main_frame.rowconfigure(6, weight=1)

        # Header
        ttk.Label(self.main_frame, text="CiteQuery: Accurate AI for Government Data", style='Header.TLabel').grid(row=0, column=0, pady=10, sticky=tk.W)

        # Create a frame for the dataset info with scrollbar
        info_frame = ttk.Frame(self.main_frame)
        info_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        # Dataset info text with scrollbar
        self.dataset_info_text = tk.Text(info_frame, height=8, width=80, font=('Segoe UI', 12), bg='#E3ECF7', fg=text_color, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.dataset_info_text.yview)
        self.dataset_info_text.configure(yscrollcommand=scrollbar.set)

        self.dataset_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Query input
        ttk.Label(self.main_frame, text="Your Question:", style='TLabel').grid(row=2, column=0, sticky=tk.W)
        self.query_entry = ttk.Entry(self.main_frame, width=70, font=('Segoe UI', 13))
        self.query_entry.grid(row=2, column=1, pady=5, sticky=(tk.W, tk.E))

        # Suggest Question button
        ttk.Button(self.main_frame, text="Suggest Question", command=self.suggest_questions).grid(row=3, column=1, pady=5, sticky=tk.W)

        # Suggested questions
        ttk.Label(self.main_frame, text="Suggested Questions:", style='TLabel').grid(row=4, column=0, sticky=tk.W)
        self.suggested_questions = tk.Listbox(self.main_frame, height=6, font=('Segoe UI', 12))
        self.suggested_questions.grid(row=4, column=1, pady=5, sticky=(tk.W, tk.E))
        self.suggested_questions.bind('<<ListboxSelect>>', self.use_suggested_question)

        # Suggested query
        ttk.Label(self.main_frame, text="Suggested SQL:", style='TLabel').grid(row=5, column=0, sticky=tk.W)
        self.query_text = tk.Text(self.main_frame, height=3, width=60, font=('Segoe UI', 12), bg='#E3ECF7', fg=text_color, wrap=tk.WORD)
        self.query_text.grid(row=5, column=1, pady=5, sticky=(tk.W, tk.E))

        # Results section - modify to include trust box, table, and chart
        ttk.Label(self.main_frame, text="Results:", style='TLabel').grid(row=6, column=0, sticky=tk.W)
        
        # Create a frame to hold trust box, table, and chart
        results_frame = ttk.Frame(self.main_frame)
        results_frame.grid(row=6, column=1, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=0)  # Trust box fixed width (left)
        results_frame.columnconfigure(1, weight=2)  # Table gets more space (middle)
        results_frame.columnconfigure(2, weight=1)  # Chart gets remaining space (right)
        results_frame.rowconfigure(0, weight=1)
        
        # Trust Score Box (left side)
        trust_frame = ttk.LabelFrame(results_frame, text="Trust Score", padding="10")
        trust_frame.grid(row=0, column=0, pady=5, padx=(0, 5), sticky=(tk.N, tk.S))
        trust_frame.configure(width=150)
        trust_frame.grid_propagate(False)
        
        # Trust score display
        self.trust_score_label = tk.Label(trust_frame, text="N/A", font=('Segoe UI', 24, 'bold'), 
                                         bg='white', fg='#666666', width=6, height=2, 
                                         relief=tk.RAISED, borderwidth=2)
        self.trust_score_label.pack(pady=(5, 10))
        
        # Trust score status
        self.trust_status_label = tk.Label(trust_frame, text="No Query", font=('Segoe UI', 11, 'bold'), 
                                          bg='white', fg='#666666', wraplength=120)
        self.trust_status_label.pack(pady=(0, 5))
        
        # Trust score description
        self.trust_desc_label = tk.Label(trust_frame, text="Execute a query to see trust score", 
                                        font=('Segoe UI', 9), bg='white', fg='#888888', 
                                        wraplength=120, justify=tk.CENTER)
        self.trust_desc_label.pack(pady=(0, 5))
        
        # Progress bar for trust score
        self.trust_progress = ttk.Progressbar(trust_frame, length=120, mode='determinate')
        self.trust_progress.pack(pady=(5, 10))
        
        # Table frame with scrollbars (middle)
        table_frame = ttk.Frame(results_frame)
        table_frame.grid(row=0, column=1, pady=5, padx=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        
        # Create Treeview for table display
        self.result_table = ttk.Treeview(table_frame, show='headings', height=10)
        self.result_table.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Vertical scrollbar for table
        v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.result_table.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.result_table.configure(yscrollcommand=v_scrollbar.set)
        
        # Horizontal scrollbar for table
        h_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.result_table.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.result_table.configure(xscrollcommand=h_scrollbar.set)
        
        # Status text area (under table)
        self.result_text = tk.Text(table_frame, height=2, width=40, font=('Segoe UI', 10), bg='#E3ECF7', fg=text_color, wrap=tk.WORD)
        self.result_text.grid(row=2, column=0, columnspan=2, pady=(5, 0), sticky=(tk.W, tk.E))
        
        # Chart frame (right side)
        self.chart_frame = ttk.Frame(results_frame)
        self.chart_frame.grid(row=0, column=2, pady=5, padx=(5, 0), sticky=(tk.W, tk.E, tk.N, tk.S))

        # Ethical note
        ethical_note = "Note: This tool ensures ethical AI practices including privacy protection and bias prevention. All queries are audited."
        ttk.Label(self.main_frame, text=ethical_note, font=('Segoe UI', 11, 'italic'), background=main_bg, foreground=text_color).grid(row=7, column=0, columnspan=2, pady=10, sticky=tk.W)
        
        # Add status bar at the bottom
        self.status_frame = ttk.Frame(self.main_frame, relief=tk.SUNKEN, borderwidth=1)
        self.status_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        self.status_frame.columnconfigure(1, weight=1)
        
        ttk.Label(self.status_frame, text="Status:", font=('Segoe UI', 11, 'bold')).grid(row=0, column=0, padx=5)
        self.status_bar_label = ttk.Label(self.status_frame, text="Ready", font=('Segoe UI', 11))
        self.status_bar_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(self.status_frame, text="Trust:", font=('Segoe UI', 11, 'bold')).grid(row=0, column=2, padx=5)
        self.trust_bar_label = ttk.Label(self.status_frame, text="N/A", font=('Segoe UI', 11, 'bold'))
        self.trust_bar_label.grid(row=0, column=3, padx=5)

    def upload_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV/Excel Files", "*.csv *.xlsx")])
        if file_path:
            dataset_name = self.chatbot.load_dataset(file_path)
            if dataset_name:
                self.datasets_list.append(dataset_name)
                self.dataset_combo['values'] = self.datasets_list
                self.dataset_combo.set(dataset_name)
                self.current_dataset = dataset_name
                self.display_dataset_info()
                self.suggested_questions.delete(0, tk.END)  # Clear suggestions on dataset load
                messagebox.showinfo("Success", f"Dataset '{dataset_name}' loaded successfully.")
            else:
                messagebox.showerror("Error", "Failed to load dataset.")

    def select_dataset(self, event):
        self.current_dataset = self.dataset_combo.get()
        self.display_dataset_info()
        self.suggested_questions.delete(0, tk.END)  # Clear suggestions on dataset change

    def display_dataset_info(self):
        if not self.current_dataset:
            return
        info = self.chatbot.get_dataset_info(self.current_dataset)
        self.dataset_info_text.delete(1.0, tk.END)
        self.dataset_info_text.insert(tk.END, f"Dataset: {self.current_dataset}\nColumns: {', '.join(info['columns'])}\n")
        self.dataset_info_text.insert(tk.END, "Statistics:\n")
        for col, stats in info['stats'].items():
            self.dataset_info_text.insert(tk.END, f"{col}: Mean={stats.get('mean', 0):.2f}, Min={stats.get('min', 0)}, Max={stats.get('max', 0)}\n")

    def suggest_questions(self):
        if not self.current_dataset:
            self.suggested_questions.delete(0, tk.END)
            messagebox.showerror("Error", "No dataset selected.")
            return
        user_input = self.query_entry.get()
        if not user_input:
            self.suggested_questions.delete(0, tk.END)
            messagebox.showerror("Error", "Enter a question to get suggestions.")
            return
        all_suggestions = [
            "What are the average days taken by department?",
            "Leave patterns by month.",
            "Total leave days in Finance department",
            "Average remaining leaves in Operations",
            "Maternity leave count by department",
        ]
        lemmatized_sugs = [lemmatize_sentence(s) for s in all_suggestions]
        vectorizer = TfidfVectorizer()
        vectorizer.fit(lemmatized_sugs + [lemmatize_sentence(user_input)])
        input_vec = vectorizer.transform([lemmatize_sentence(user_input)])
        sugs_vec = vectorizer.transform(lemmatized_sugs)
        scores = cosine_similarity(input_vec, sugs_vec)[0]
        sorted_indices = np.argsort(scores)[::-1]
        top_sugs = [all_suggestions[i] for i in sorted_indices if scores[i] > 0.1][:5]
        self.suggested_questions.delete(0, tk.END)
        for sug in top_sugs:
            self.suggested_questions.insert(tk.END, sug)

    def use_suggested_question(self, event):
        selection = self.suggested_questions.curselection()
        if selection:
            sug = self.suggested_questions.get(selection[0])
            self.query_entry.delete(0, tk.END)
            self.query_entry.insert(0, sug)

    def suggest_query(self):
        if not self.current_dataset:
            messagebox.showerror("Error", "No dataset selected.")
            return
        user_input = self.query_entry.get()
        if not user_input:
            messagebox.showerror("Error", "Enter a question.")
            return
        query, params, assumption = self.chatbot.suggest_query(user_input, self.current_dataset)
        self.suggested_query = query
        self.suggested_params = params
        self.query_text.delete(1.0, tk.END)
        self.query_text.insert(tk.END, f"Suggested Query: {query}\nParameters: {params}\n{assumption}")

    def run_query(self):
        if not self.suggested_query:
            messagebox.showerror("Error", "No query suggested.")
            return
        
        result, message, trust_score = self.chatbot.execute_query(self.suggested_query, self.suggested_params)
        
        # Update trust score box with enhanced styling
        self.update_trust_score_display(trust_score, message)
        
        # Update status text
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Status: {message}\n")
        self.result_text.insert(tk.END, f"Rows: {len(result) if not result.empty else 0}")
        
        # Update status bar (keep existing)
        self.status_bar_label.config(text=f"{message} | Rows: {len(result) if not result.empty else 0}")
        if trust_score >= 0.8:
            trust_color = "#28A745"
        elif trust_score >= 0.6:
            trust_color = "#FFC107"
        else:
            trust_color = "#DC3545"
        self.trust_bar_label.config(text=f"{trust_score:.2f}", foreground=trust_color)
        
        if not result.empty:
            self.populate_table(result)
            self.generate_chart(result)
        else:
            self.clear_table()
            self.clear_chart()

    def update_trust_score_display(self, trust_score, message):
        """Update the trust score display box with colors and status."""
        # Update trust score number
        self.trust_score_label.config(text=f"{trust_score:.2f}")
        
        # Update progress bar
        self.trust_progress['value'] = trust_score * 100
        
        # Color coding and status based on trust score
        if trust_score >= 0.8:
            # High trust - Green
            bg_color = "#D4EFDF"  # Light green
            text_color = "#27AE60"  # Dark green
            status_text = "HIGH TRUST"
            desc_text = "Data is highly reliable and accurate"
            progress_style = "TProgressbar"
        elif trust_score >= 0.6:
            # Medium trust - Yellow/Orange
            bg_color = "#FCF3CF"  # Light yellow
            text_color = "#F39C12"  # Orange
            status_text = "MEDIUM TRUST"
            desc_text = "Data is moderately reliable, verify if needed"
            progress_style = "TProgressbar"
        elif trust_score >= 0.3:
            # Low trust - Red
            bg_color = "#FADBD8"  # Light red
            text_color = "#E74C3C"  # Dark red
            status_text = "LOW TRUST"
            desc_text = "Data reliability is questionable, manual verification recommended"
            progress_style = "TProgressbar"
        else:
            # Very low trust - Dark red
            bg_color = "#F1948A"  # Darker light red
            text_color = "#C0392B"  # Very dark red
            status_text = "VERY LOW"
            desc_text = "Data is unreliable, do not use without verification"
            progress_style = "TProgressbar"
        
        # Apply colors to trust score display
        self.trust_score_label.config(bg=bg_color, fg=text_color, relief=tk.RAISED, borderwidth=3)
        self.trust_status_label.config(text=status_text, fg=text_color, bg='white')
        self.trust_desc_label.config(text=desc_text, fg=text_color, bg='white')
        
        # Update progress bar color (if possible with your theme)
        style = ttk.Style()
        if trust_score >= 0.8:
            style.configure("TProgressbar", background='#27AE60')
        elif trust_score >= 0.6:
            style.configure("TProgressbar", background='#F39C12')
        else:
            style.configure("TProgressbar", background='#E74C3C')

    def clear_trust_display(self):
        """Reset trust score display to default state."""
        self.trust_score_label.config(text="N/A", bg='white', fg='#666666')
        self.trust_status_label.config(text="No Query", fg='#666666')
        self.trust_desc_label.config(text="Execute a query to see trust score", fg='#888888')
        self.trust_progress['value'] = 0

    def populate_table(self, data):
        """Populate the Treeview table with data."""
        # Clear existing data
        self.clear_table()
        
        # Configure columns
        columns = list(data.columns)
        self.result_table['columns'] = columns
        
        # Configure column headings and widths
        for col in columns:
            self.result_table.heading(col, text=col.replace('_', ' ').title())
            # Auto-adjust column width based on content
            max_width = max(
                len(col.replace('_', ' ').title()) * 10,  # Header width
                max([len(str(data[col].iloc[i])) * 8 for i in range(min(len(data), 10))]) if len(data) > 0 else 100  # Content width (check first 10 rows)
            )
            self.result_table.column(col, width=min(max_width, 200), anchor='center')
        
        # Insert data rows
        for index, row in data.iterrows():
            # Format numeric values to 2 decimal places if they're floats
            formatted_row = []
            for value in row:
                if isinstance(value, float):
                    formatted_row.append(f"{value:.2f}")
                else:
                    formatted_row.append(str(value))
            
            self.result_table.insert('', 'end', values=formatted_row)
        
        # Style the table
        style = ttk.Style()
        style.configure("Treeview.Heading", font=('Segoe UI', 11, 'bold'), background='#1A73E8', foreground='white')
        style.configure("Treeview", font=('Segoe UI', 10), rowheight=25, background='#F8FAFC', foreground='#222222')
        style.map('Treeview', background=[('selected', '#E3ECF7')])

    def clear_table(self):
        """Clear the table contents."""
        for item in self.result_table.get_children():
            self.result_table.delete(item)

    def generate_chart(self, data):
        """Generate a bar chart for suitable data."""
        try:
            # Clear previous chart
            self.clear_chart()
            
            # Check if data is suitable for charting
            if len(data.columns) != 2:
                return  # Need exactly 2 columns for simple bar chart
            
            col1, col2 = data.columns
            
            # Check if we have categorical data in first column and numeric in second
            if not (data[col1].dtype == 'object' and np.issubdtype(data[col2].dtype, np.number)):
                return
            
            # Create matplotlib figure with a modern style
            style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
            
            # Create bar chart
            bars = ax.bar(data[col1], data[col2], color='#1A73E8', alpha=0.8, edgecolor='#0F4C8C', linewidth=1)
            
            # Customize the chart
            ax.set_xlabel(col1.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            ax.set_ylabel(col2.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            ax.set_title(f'{col2.replace("_", " ").title()} by {col1.replace("_", " ").title()}', 
                        fontsize=12, fontweight='bold', pad=20)
            
            # Rotate x-axis labels if they're long
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            
            # Improve layout
            plt.tight_layout()
            
            # Embed the chart in Tkinter
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Store canvas reference to prevent garbage collection
            self.current_canvas = canvas
            
        except Exception as e:
            print(f"Chart generation failed: {e}")
            # If chart generation fails, just continue without chart

    def clear_chart(self):
        """Clear the chart area."""
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        if hasattr(self, 'current_canvas'):
            delattr(self, 'current_canvas')

    def view_audit_log(self):
        with open('chatbot_audit.log', 'r') as f:
            log_content = f.read()
        log_window = tk.Toplevel(self.root)
        log_window.title("Audit Log")
        log_text = tk.Text(log_window, wrap=tk.WORD, font=('Segoe UI', 12))
        log_text.insert(tk.END, log_content)
        log_text.pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = CiteQueryApp(root)
    root.mainloop()