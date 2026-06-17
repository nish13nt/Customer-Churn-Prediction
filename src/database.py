import mysql.connector
from mysql.connector import Error
import sys
import os

# Add the parent directory to the system path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def create_db_connection(host_name, user_name, user_password, db_name=None):
    """Establishes a connection to the MySQL server."""
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name,
            auth_plugin='mysql_native_password'
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")
    return connection

def execute_query(connection, query):
    """Executes a single SQL query."""
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query executed successfully")
    except Error as err:
        print(f"Error: '{err}'")

def initialize_database():
    """Creates the schema, tables, and relationships."""
    # 1. Connect without specifying a DB to create the DB first
    conn = create_db_connection(config.DB_HOST, config.DB_USER, config.DB_PASSWORD)
    
    if conn is not None:
        create_db_query = f"CREATE DATABASE IF NOT EXISTS {config.DB_NAME};"
        execute_query(conn, create_db_query)
        conn.close()

    # 2. Reconnect with the newly created DB
    conn = create_db_connection(config.DB_HOST, config.DB_USER, config.DB_PASSWORD, config.DB_NAME)
    
    if conn is not None:
        # 3. Define Table Schemas (Normalized 3NF)
        create_customers_table = """
        CREATE TABLE IF NOT EXISTS customers (
            customer_id VARCHAR(20) PRIMARY KEY,
            gender VARCHAR(10),
            senior_citizen INT,
            partner VARCHAR(3),
            dependents VARCHAR(3)
        );
        """
        
        create_financials_table = """
        CREATE TABLE IF NOT EXISTS financials (
            customer_id VARCHAR(20) PRIMARY KEY,
            monthly_charges DECIMAL(10,2),
            total_charges DECIMAL(10,2),
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
        );
        """

        create_predictions_table = """
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INT AUTO_INCREMENT PRIMARY KEY,
            customer_id VARCHAR(20),
            churn_risk_score INT,
            action_recommendation VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
        );
        """
        
        # 4. Create Indexes for optimization
        create_index_query = "CREATE INDEX idx_risk_score ON predictions(churn_risk_score);"

        # Execute all queries
        print("Creating tables...")
        execute_query(conn, create_customers_table)
        execute_query(conn, create_financials_table)
        execute_query(conn, create_predictions_table)
        
        # We wrap index creation in a try-except block because MySQL throws an error if it already exists
        try:
            execute_query(conn, create_index_query)
        except:
            print("Index likely already exists, skipping.")
            
        conn.close()

if __name__ == "__main__":
    initialize_database()