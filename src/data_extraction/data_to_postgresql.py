import pandas as pd
from sqlalchemy import create_engine

def dataframe_to_postgres(df, table_name, database_url, primary_key=None):
    """
    Convert a pandas DataFrame to a PostgreSQL table.

    Parameters:
    - df (pandas.DataFrame): DataFrame to be inserted into PostgreSQL table.
    - table_name (str): Name of the table in PostgreSQL.
    - database_url (str): Connection string for the PostgreSQL database.
    - primary_key (str, optional): Column to set as the primary key.
    """
    engine = create_engine(database_url)
    
    if primary_key:
        # Set the DataFrame index to the primary key
        df = df.set_index(primary_key)
    
    # Use 'replace' if you want to overwrite existing tables, 'append' to add to them, or 'fail' to do nothing if the table exists
    df.to_sql(table_name, engine, if_exists='replace', index=True, index_label=primary_key)
    print(f"DataFrame has been written to '{table_name}' in the PostgreSQL database with primary key: {primary_key}")

if __name__ == "__main__":
    # Sample DataFrame
    df = pd.DataFrame({
        'id': [101, 102, 103],
        'a': [1, 2, 3],
        'b': ['x', 'y', 'z']
    })
    
    # Database connection string
    DATABASE_URL = "postgresql://username:password@localhost:5432/mydatabase"
    TABLE_NAME = "sample_table"
    PRIMARY_KEY = "id"
    
    dataframe_to_postgres(df, TABLE_NAME, DATABASE_URL, PRIMARY_KEY)
