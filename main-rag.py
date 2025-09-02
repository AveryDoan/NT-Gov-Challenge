import pandas as pd
import sqlite3
import re
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import os
from datetime import datetime
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize_sentence(sentence):
    return ' '.join([lemmatizer.lemmatize(word) for word in sentence.lower().split()])

# Configure logging for audit trails
logging.basicConfig(filename='chatbot_audit.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class APIConfigDialog:
    def __init__(self, parent=None):
        self.api_url = None
        self.api_key = None
        self.result = False
        
        # Create the dialog window
        self.dialog = tk.Toplevel(parent) if parent else tk.Tk()
        self.dialog.title("CiteQuery - API Configuration")
        self.dialog.geometry("500x300")
        self.dialog.resizable(False, False)
        
        # Center the window
        self.dialog.geometry("+{}+{}".format(
            (self.dialog.winfo_screenwidth() // 2) - 250,
            (self.dialog.winfo_screenheight() // 2) - 150
        ))
        
        # Make it modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Configure styles
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'))
        style.configure('Config.TLabel', font=('Segoe UI', 12))
        style.configure('Config.TEntry', font=('Segoe UI', 11), padding=5)
        style.configure('Config.TButton', font=('Segoe UI', 11, 'bold'), padding=8)
        
        self.create_widgets()
        
        # Load saved settings if they exist
        self.load_saved_settings()
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="🔧 API Configuration", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Description
        desc_text = "Configure your API settings to connect to external services.\nThis includes LLM APIs like OpenAI, Hugging Face, or Azure."
        desc_label = ttk.Label(main_frame, text=desc_text, style='Config.TLabel', justify=tk.CENTER)
        desc_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # API URL
        ttk.Label(main_frame, text="API Base URL:", style='Config.TLabel').grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        self.url_entry = ttk.Entry(main_frame, width=50, style='Config.TEntry')
        self.url_entry.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        self.url_entry.insert(0, "https://api.openai.com/v1")  # Default
        
        # API Key
        ttk.Label(main_frame, text="API Key:", style='Config.TLabel').grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        self.key_entry = ttk.Entry(main_frame, width=50, style='Config.TEntry', show="*")
        self.key_entry.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Show/Hide key button
        self.show_key_var = tk.BooleanVar()
        show_key_check = ttk.Checkbutton(main_frame, text="Show API Key", variable=self.show_key_var, command=self.toggle_key_visibility)
        show_key_check.grid(row=6, column=0, sticky=tk.W, pady=(0, 20))
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=2, pady=(10, 0))
        
        # Buttons
        ttk.Button(button_frame, text="Test Connection", command=self.test_connection, style='Config.TButton').grid(row=0, column=0, padx=(0, 10))
        ttk.Button(button_frame, text="Save & Continue", command=self.save_and_continue, style='Config.TButton').grid(row=0, column=1, padx=(0, 10))
        ttk.Button(button_frame, text="Skip", command=self.skip_config, style='Config.TButton').grid(row=0, column=2)
        
        # Make columns expandable
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
    def toggle_key_visibility(self):
        if self.show_key_var.get():
            self.key_entry.config(show="")
        else:
            self.key_entry.config(show="*")
            
    def test_connection(self):
        """Test the API connection (placeholder)"""
        url = self.url_entry.get().strip()
        key = self.key_entry.get().strip()
        
        if not url or not key:
            messagebox.showerror("Error", "Please enter both API URL and API Key.", parent=self.dialog)
            return
            
        # Placeholder for actual API testing
        # You can implement actual API testing here
        try:
            # Simulate API test
            messagebox.showinfo("Success", "API connection test successful!\n(This is a simulation)", parent=self.dialog)
        except Exception as e:
            messagebox.showerror("Error", f"API connection failed:\n{str(e)}", parent=self.dialog)
    
    def save_and_continue(self):
        """Save the configuration and continue"""
        self.api_url = self.url_entry.get().strip()
        self.api_key = self.key_entry.get().strip()
        
        if not self.api_url:
            messagebox.showerror("Error", "Please enter an API URL.", parent=self.dialog)
            return
            
        # Save to file for next time
        self.save_settings()
        
        self.result = True
        self.dialog.destroy()
        
    def skip_config(self):
        """Skip API configuration"""
        result = messagebox.askyesno("Skip Configuration", 
                                   "Are you sure you want to skip API configuration?\n\nSome features may not work without proper API setup.", 
                                   parent=self.dialog)
        if result:
            self.api_url = None
            self.api_key = None
            self.result = True
            self.dialog.destroy()
    
    def save_settings(self):
        """Save settings to a config file"""
        try:
            config_data = f"API_URL={self.api_url}\nAPI_KEY={self.api_key}\n"
            with open("config.txt", "w") as f:
                f.write(config_data)
            logging.info("API configuration saved")
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
    
    def load_saved_settings(self):
        """Load previously saved settings"""
        try:
            if os.path.exists("config.txt"):
                with open("config.txt", "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("API_URL="):
                            saved_url = line.replace("API_URL=", "").strip()
                            if saved_url:
                                self.url_entry.delete(0, tk.END)
                                self.url_entry.insert(0, saved_url)
                        elif line.startswith("API_KEY="):
                            saved_key = line.replace("API_KEY=", "").strip()
                            if saved_key:
                                self.key_entry.delete(0, tk.END)
                                self.key_entry.insert(0, saved_key)
        except Exception as e:
            logging.error(f"Failed to load config: {e}")

class CiteQueryChatbot:
    def __init__(self, api_url=None, api_key=None):
        """Initialize the chatbot with optional API configuration."""
        self.api_url = api_url
        self.api_key = api_key
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
        logging.info(f"CiteQueryChatbot initialized with API: {api_url is not None}")

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
        }

        if 'employee_leave' in dataset_name or 'leave' in dataset_name:
            templates.update({
                "leave_patterns": f"SELECT department, leave_type, AVG(days_taken) as avg_days, COUNT(*) as count, SUM(days_taken) as total_days FROM {dataset_name} GROUP BY department, leave_type",
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
    def __init__(self, root, api_url=None, api_key=None):
        self.root = root
        self.root.title("CiteQuery Dashboard - NT Government AI Solution")
        self.chatbot = CiteQueryChatbot(api_url, api_key)
        self.current_dataset = None
        self.suggested_query = None
        self.suggested_params = None
        self.datasets_list = []

        # Display API status in title if configured
        if api_url:
            self.root.title("CiteQuery Dashboard - NT Government AI Solution [API Connected]")
        
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

        # Dataset info
        ttk.Label(self.main_frame, text="Dataset Info:", style='TLabel').grid(row=0, column=1, sticky=tk.W)
        self.dataset_info_text = tk.Text(self.main_frame, height=4, width=60, font=('Segoe UI', 12), bg='#E3ECF7', fg=text_color, wrap=tk.WORD)
        self.dataset_info_text.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

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

        # Results
        ttk.Label(self.main_frame, text="Results:", style='TLabel').grid(row=6, column=0, sticky=tk.W)
        self.result_text = tk.Text(self.main_frame, height=8, width=60, font=('Segoe UI', 12), bg='#E3ECF7', fg=text_color, wrap=tk.WORD)
        self.result_text.grid(row=6, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        # Ethical note
        ethical_note = "Note: This tool ensures ethical AI practices including privacy protection and bias prevention. All queries are audited."
        ttk.Label(self.main_frame, text=ethical_note, font=('Segoe UI', 11, 'italic'), background=main_bg, foreground=text_color).grid(row=7, column=0, columnspan=2, pady=10, sticky=tk.W)

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
                messagebox.showinfo("Success", f"Dataset '{dataset_name}' loaded successfully.")
            else:
                messagebox.showerror("Error", "Failed to load dataset.")

    def select_dataset(self, event):
        self.current_dataset = self.dataset_combo.get()
        self.display_dataset_info()

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
            return
        user_input = self.query_entry.get()
        all_suggestions = [
            "What are the average days taken by department?", 
            "Leave patterns by month.", 
            "Total leave days in Finance department",
            "Average remaining leaves in Operations",
            "Maternity leave count by department"
        ]
        if not user_input:
            top_sugs = all_suggestions[:5]
        else:
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
        self.suggest_questions()  # Populate suggested questions when Suggest Query is clicked

    def run_query(self):
        if not self.suggested_query:
            messagebox.showerror("Error", "No query suggested.")
            return
        result, message, trust_score = self.chatbot.execute_query(self.suggested_query, self.suggested_params)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"{message} (Trust: {trust_score:.2f})\n\n")
        if not result.empty:
            self.result_text.insert(tk.END, result.to_string(index=False))
        else:
            self.result_text.insert(tk.END, "No data returned.")

    def view_audit_log(self):
        with open('chatbot_audit.log', 'r') as f:
            log_content = f.read()
        log_window = tk.Toplevel(self.root)
        log_window.title("Audit Log")
        log_text = tk.Text(log_window, wrap=tk.WORD, font=('Segoe UI', 12))
        log_text.insert(tk.END, log_content)
        log_text.pack(fill=tk.BOTH, expand=True)

def main():
    """Main function with API configuration dialog"""
    # Show API configuration dialog first
    config_dialog = APIConfigDialog()
    config_dialog.dialog.mainloop()
    
    if not config_dialog.result:
        # User closed the dialog without configuring
        return
    
    # Create main application window
    root = tk.Tk()
    
    # Initialize app with API configuration
    app = CiteQueryApp(root, config_dialog.api_url, config_dialog.api_key)
    
    # Show welcome message with API status
    if config_dialog.api_url and config_dialog.api_key:
        messagebox.showinfo("Welcome", "CiteQuery initialized successfully!\n\nAPI configuration loaded.", parent=root)
    else:
        messagebox.showinfo("Welcome", "CiteQuery initialized!\n\nRunning in offline mode.", parent=root)
    
    root.mainloop()

if __name__ == "__main__":
    main()
