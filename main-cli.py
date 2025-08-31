import pandas as pd
import sqlite3
import re
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os
from datetime import datetime
import nltk
from nltk.stem import WordNetLemmatizer

# Download NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
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

def print_banner():
    """Print the application banner."""
    print("=" * 80)
    print("🎯 CiteQuery Dashboard - NT Government AI Solution")
    print("=" * 80)
    print("AI-powered chatbot for government data analysis")
    print("Built for the NT Government Hackathon 2025")
    print("=" * 80)
    print()

def print_menu():
    """Print the main menu."""
    print("\n📋 Available Commands:")
    print("1.  load <filename>     - Load a dataset (CSV or Excel)")
    print("2.  list                - List loaded datasets")
    print("3.  info <dataset>      - Show dataset information")
    print("4.  ask <question>      - Ask a question about your data")
    print("5.  suggest             - Get suggested questions")
    print("6.  execute             - Execute the last suggested query")
    print("7.  history             - Show query history")
    print("8.  audit               - View audit log")
    print("9.  help                - Show this help menu")
    print("10. quit                - Exit the application")
    print()

def main():
    """Main CLI application."""
    print_banner()
    
    chatbot = CiteQueryChatbot()
    current_dataset = None
    suggested_query = None
    suggested_params = None
    
    print("🚀 Welcome to CiteQuery! Type 'help' to see available commands.")
    print("💡 Try loading a dataset first: load Data/employee leave tracking data.csv")
    print()
    
    while True:
        try:
            command = input("CiteQuery> ").strip()
            
            if not command:
                continue
                
            parts = command.split(' ', 1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if cmd == 'quit' or cmd == 'exit':
                print("👋 Goodbye! Thank you for using CiteQuery.")
                chatbot.close()
                break
                
            elif cmd == 'help':
                print_menu()
                
            elif cmd == 'load':
                if not args:
                    print("❌ Please specify a filename to load.")
                    print("💡 Example: load Data/employee leave tracking data.csv")
                    continue
                    
                file_path = args.strip()
                if not os.path.exists(file_path):
                    print(f"❌ File not found: {file_path}")
                    print("💡 Available files in Data/ folder:")
                    data_files = [f for f in os.listdir('Data/') if f.endswith(('.csv', '.xlsx'))]
                    for f in data_files:
                        print(f"   - Data/{f}")
                    continue
                    
                print(f"📂 Loading dataset: {file_path}")
                dataset_name = chatbot.load_dataset(file_path)
                if dataset_name:
                    current_dataset = dataset_name
                    print(f"✅ Dataset '{dataset_name}' loaded successfully!")
                    print(f"📊 Columns: {', '.join(chatbot.get_dataset_info(dataset_name)['columns'])}")
                else:
                    print("❌ Failed to load dataset.")
                    
            elif cmd == 'list':
                if not chatbot.datasets:
                    print("📭 No datasets loaded. Use 'load <filename>' to load a dataset.")
                else:
                    print("📚 Loaded Datasets:")
                    for name, df in chatbot.datasets.items():
                        print(f"   - {name}: {len(df)} rows, {len(df.columns)} columns")
                        
            elif cmd == 'info':
                if not args:
                    if not current_dataset:
                        print("❌ No dataset selected. Use 'load <filename>' first.")
                        continue
                    dataset_name = current_dataset
                else:
                    dataset_name = args.strip()
                    
                if dataset_name not in chatbot.datasets:
                    print(f"❌ Dataset '{dataset_name}' not found.")
                    continue
                    
                info = chatbot.get_dataset_info(dataset_name)
                print(f"📊 Dataset: {dataset_name}")
                print(f"📋 Columns: {', '.join(info['columns'])}")
                print("📈 Statistics:")
                for col, stats in info['stats'].items():
                    if isinstance(stats, dict):
                        print(f"   {col}: Mean={stats.get('mean', 0):.2f}, Min={stats.get('min', 0)}, Max={stats.get('max', 0)}")
                        
            elif cmd == 'ask':
                if not args:
                    print("❌ Please provide a question.")
                    print("💡 Example: ask 'How many days each department take in average?'")
                    continue
                    
                if not current_dataset:
                    print("❌ No dataset selected. Use 'load <filename>' first.")
                    continue
                    
                question = args.strip()
                print(f"🤔 Processing question: {question}")
                
                query, params, assumption = chatbot.suggest_query(question, current_dataset)
                suggested_query = query
                suggested_params = params
                
                print(f"🔍 Suggested SQL: {query}")
                print(f"📝 Parameters: {params}")
                print(f"💭 Assumption: {assumption}")
                print("\n💡 Use 'execute' to run this query or ask another question.")
                
            elif cmd == 'suggest':
                if not current_dataset:
                    print("❌ No dataset selected. Use 'load <filename>' first.")
                    continue
                    
                print("💡 Suggested Questions:")
                suggestions = [
                    "How many days each department take in average?",
                    "Average leave days in HR department?",
                    "Total leave taken in IT department.",
                    "Leave patterns by month.",
                    "Summary statistics for the dataset.",
                    "Total leave days in Finance department",
                    "How many employees in Marketing took casual leave?",
                    "Average remaining leaves in Operations",
                    "List employees with zero remaining leaves",
                    "Maternity leave count by department"
                ]
                
                for i, suggestion in enumerate(suggestions[:5], 1):
                    print(f"   {i}. {suggestion}")
                print("\n💡 Use 'ask <question>' to ask any of these questions.")
                
            elif cmd == 'execute':
                if not suggested_query:
                    print("❌ No query to execute. Use 'ask <question>' first.")
                    continue
                    
                print(f"🚀 Executing query: {suggested_query}")
                print(f"📝 Parameters: {suggested_params}")
                
                result, message, trust_score = chatbot.execute_query(suggested_query, suggested_params)
                
                print(f"📊 Result: {message}")
                print(f"🔒 Trust Score: {trust_score:.2f}")
                
                if not result.empty:
                    print("\n📋 Data:")
                    print(result.to_string(index=False))
                else:
                    print("📭 No data returned.")
                    
            elif cmd == 'history':
                if not chatbot.query_history:
                    print("📭 No query history available.")
                else:
                    print("📚 Query History:")
                    for i, (question, query, params) in enumerate(chatbot.query_history[-5:], 1):
                        print(f"   {i}. Q: {question}")
                        print(f"      SQL: {query}")
                        print(f"      Params: {params}")
                        print()
                        
            elif cmd == 'audit':
                try:
                    with open('chatbot_audit.log', 'r') as f:
                        log_content = f.read()
                    print("📝 Audit Log:")
                    print(log_content[-1000:])  # Show last 1000 characters
                except FileNotFoundError:
                    print("📝 No audit log found.")
                    
            else:
                print(f"❌ Unknown command: {cmd}")
                print("💡 Type 'help' to see available commands.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thank you for using CiteQuery.")
            chatbot.close()
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("💡 Type 'help' for assistance.")

if __name__ == "__main__":
    main()
