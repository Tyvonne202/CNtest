import logging
import psycopg2
import pandas as pd
import json
import openai  # Assuming you have the GPT-4 library installed and API key set up



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MyApplication')

db_conn = None

def get_db_connection():
    global db_conn
    if db_conn is None or db_conn.closed:
        try:
            db_conn = psycopg2.connect(
                database="postgres",
                user="postgres",
                password="Vbevbe1!",  # Replace with your password
                host="localhost",
                port="5432"
            )
            logger.info("Successfully connected to the database.")
        except Exception as e:
            logger.error(f"Failed to connect to the database. Error: {str(e)}")
            return None
    return db_conn


def get_table_layout(conn):
    cursor = conn.cursor()

    # Define queries to get table layouts and column comments
    query_table_layouts = """
    SELECT table_name, column_name, data_type 
    FROM information_schema.columns 
    WHERE table_schema = 'public';
    """

    query_column_comments = """
    SELECT cols.table_name, cols.column_name, (
        SELECT pg_catalog.col_description(c.oid, cols.ordinal_position::int)
        FROM pg_catalog.pg_class c
        WHERE c.oid = (SELECT ('"' || cols.table_name || '"')::regclass::oid)
        AND c.relname = cols.table_name
    ) AS column_comment
    FROM information_schema.columns cols
    WHERE cols.table_schema = 'public';
    """

    # Execute queries and fetch results
    cursor.execute(query_table_layouts)
    table_layouts = cursor.fetchall()

    cursor.execute(query_column_comments)
    column_comments = cursor.fetchall()

    return table_layouts, column_comments


def create_schema(conn):
    cursor = conn.cursor()

    # List of SQL commands to execute
    commands = [
        """
        CREATE TABLE roles (
            role_id UUID PRIMARY KEY,
            role_name TEXT NOT NULL UNIQUE
        );
        """,
        """
        CREATE TABLE permissions (
            permission_id UUID PRIMARY KEY,
            permission_name TEXT NOT NULL UNIQUE
        );
        """,
        """
        CREATE TABLE users (
            user_id UUID PRIMARY KEY,
            username TEXT NOT NULL,
            user_info JSONB,
            role_id UUID REFERENCES roles(role_id),
            created_at TIMESTAMP NOT NULL,
            consent_given BOOLEAN,
            consent_timestamp TIMESTAMP,
            anonymized BOOLEAN,
            preferences JSONB
        );
        CREATE INDEX idx_users_role_id ON users(role_id);
        CREATE INDEX idx_users_created_at ON users(created_at);
        CREATE INDEX idx_user_info_key ON users USING gin(user_info);
        """,
        """
        CREATE TABLE role_permissions (
            role_id UUID REFERENCES roles(role_id),
            permission_id UUID REFERENCES permissions(permission_id)
        );
        CREATE INDEX idx_role_permissions_role_id ON role_permissions(role_id);
        CREATE INDEX idx_role_permissions_permission_id ON role_permissions(permission_id);
        """,
        """
        CREATE TABLE user_interactions (
            interaction_id UUID PRIMARY KEY,
            user_id UUID REFERENCES users(user_id),
            interaction_info JSONB,
            created_at TIMESTAMP NOT NULL
        );
        CREATE INDEX idx_user_interactions_user_id ON user_interactions(user_id);
        CREATE INDEX idx_user_interactions_created_at ON user_interactions(created_at);
        CREATE INDEX idx_interaction_info_key ON user_interactions USING gin(interaction_info);
        """,
        """
        CREATE TABLE conversation_history (
            conversation_id UUID,
            user_id UUID REFERENCES users(user_id),
            interaction_id UUID REFERENCES user_interactions(interaction_id),
            message TEXT,
            response TEXT,
            timestamp TIMESTAMP NOT NULL,
            PRIMARY KEY (conversation_id, timestamp)
        )
        PARTITION BY RANGE (timestamp);
        CREATE INDEX idx_conversation_history_user_id ON conversation_history(user_id);
        CREATE INDEX idx_conversation_history_interaction_id ON conversation_history(interaction_id);
        CREATE INDEX idx_conversation_history_timestamp ON conversation_history(timestamp);
        CREATE INDEX idx_conversation_history_message_text ON conversation_history USING gin(to_tsvector('english', message));
        CREATE INDEX idx_conversation_history_response_text ON conversation_history USING gin(to_tsvector('english', response));
        """,
        """
        CREATE TABLE audit_logs_errors (
            log_id UUID PRIMARY KEY,
            user_id UUID REFERENCES users(user_id),
            action_time TIMESTAMP NOT NULL,
            action_description TEXT,
            log_type TEXT,
            error_description TEXT,
            error_location TEXT
        );
        CREATE INDEX idx_audit_logs_errors_user_id ON audit_logs_errors(user_id);
        CREATE INDEX idx_audit_logs_errors_action_time ON audit_logs_errors(action_time);
        CREATE INDEX idx_audit_logs_errors_log_type ON audit_logs_errors(log_type);
        """,
        """
        CREATE TABLE data (
            id UUID PRIMARY KEY,
            given_data_id TEXT,
            given_data JSONB,
            vbe_data JSONB,
            meta_data JSONB,
            content TEXT,
            data JSONB,
            word_count INTEGER,
            character_count INTEGER,
            classification TEXT,
            domain TEXT,
            category TEXT,
            subcategory TEXT,
            elements TEXT,
            adm TEXT,
            taxonomy_id UUID,
            knowledgebase_id UUID
        );
        CREATE INDEX idx_data_given_data_id ON data(given_data_id);
        CREATE INDEX idx_data_classification ON data(classification);
        CREATE INDEX idx_data_domain ON data(domain);
        CREATE INDEX idx_data_category ON data(category);
        CREATE INDEX idx_data_taxonomy_id ON data(taxonomy_id);
        CREATE INDEX idx_data_knowledgebase_id ON data(knowledgebase_id);
        """,
        """
        CREATE TABLE taxonomy (
            id UUID PRIMARY KEY,
            taxonomy TEXT NOT NULL,
            parent_document_id UUID,
            creation_date TIMESTAMP NOT NULL,
            last_modified_date TIMESTAMP NOT NULL,
            related_documents UUID[],
            subcategory TEXT,
            elements TEXT,
            fields TEXT,
            parent_id UUID,
            author UUID REFERENCES users(user_id),
            name TEXT,
            variable TEXT,
            file_size INTEGER,
            keywords TEXT,
            description TEXT,
            status TEXT,
            location TEXT,
            version TEXT,
            access_level TEXT,
            domain TEXT CHECK(domain IN ('Business', 'Personal', 'Objective', 'Variable')),
            classification TEXT,
            category TEXT
        );
        CREATE INDEX idx_taxonomy_domain ON taxonomy(domain);
        CREATE INDEX idx_taxonomy_classification ON taxonomy(classification);
        CREATE INDEX idx_taxonomy_category ON taxonomy(category);
        CREATE INDEX idx_taxonomy_last_modified_date ON taxonomy(last_modified_date);
        """,
        """
        CREATE TABLE taxonomy_data (
            taxonomy_id UUID REFERENCES taxonomy(id),
            data_id UUID REFERENCES data(id)
        );
        CREATE INDEX idx_taxonomy_data_taxonomy_id ON taxonomy_data(taxonomy_id);
        CREATE INDEX idx_taxonomy_data_data_id ON taxonomy_data(data_id);
        """,
        "ALTER TABLE data ADD CONSTRAINT data_taxonomy_fk FOREIGN KEY (taxonomy_id) REFERENCES taxonomy(id);",
        "ALTER TABLE taxonomy ADD CONSTRAINT taxonomy_data_fk FOREIGN KEY (parent_document_id) REFERENCES data(id);",
        """
        CREATE TABLE embeddings (
            vector_id UUID PRIMARY KEY,
            model_id UUID NOT NULL,
            created_at TIMESTAMP NOT NULL,
            metadata JSONB,
            use_count INTEGER DEFAULT 0,
            user_id UUID REFERENCES users(user_id),
            interaction_id UUID REFERENCES user_interactions(interaction_id)
        );
        CREATE INDEX idx_embeddings_vector_id ON embeddings(vector_id);
        CREATE INDEX idx_embeddings_model_id ON embeddings(model_id);
        CREATE INDEX idx_embeddings_created_at ON embeddings(created_at);
        CREATE INDEX idx_embeddings_user_id ON embeddings(user_id);
        CREATE INDEX idx_embeddings_interaction_id ON embeddings(interaction_id);
        """,
        """
        CREATE TABLE model_metrics (
            id UUID PRIMARY KEY,
            model_type TEXT,
            created_at TIMESTAMP NOT NULL,
            name TEXT,
            version TEXT,
            description TEXT,
            training_data JSONB,
            metrics JSONB,
            accuracy FLOAT,
            precision FLOAT,
            recall FLOAT,
            f1_score FLOAT,
            parent_model_id UUID REFERENCES model_metrics(id),
            embeddings_id UUID REFERENCES embeddings(vector_id)
        );
        CREATE INDEX idx_model_metrics_model_type ON model_metrics(model_type);
        CREATE INDEX idx_model_metrics_name ON model_metrics(name);
        CREATE INDEX idx_model_metrics_version ON model_metrics(version);
        CREATE INDEX idx_model_metrics_parent_model_id ON model_metrics(parent_model_id);
        CREATE INDEX idx_model_metrics_embeddings_id ON model_metrics(embeddings_id);
        """,
        """
        CREATE TABLE sessions_feedback (
            session_id UUID,
            feedback_id UUID,
            user_id UUID REFERENCES users(user_id),
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP NOT NULL,
            session_data JSONB,
            interaction_id UUID REFERENCES user_interactions(interaction_id),
            feedback TEXT,
            feedback_timestamp TIMESTAMP,
            PRIMARY KEY (session_id, feedback_id)
        );
        CREATE INDEX idx_sessions_feedback_user_id ON sessions_feedback(user_id);
        CREATE INDEX idx_sessions_feedback_start_time ON sessions_feedback(start_time);
        CREATE INDEX idx_sessions_feedback_end_time ON sessions_feedback(end_time);
        CREATE INDEX idx_sessions_feedback_interaction_id ON sessions_feedback(interaction_id);
        CREATE INDEX idx_sessions_feedback_feedback_timestamp ON sessions_feedback(feedback_timestamp);
        """,
        """
        CREATE TABLE integration_data (
            id UUID PRIMARY KEY,
            integration_type TEXT,
            integration_details JSONB,
            created_at TIMESTAMP NOT NULL,
            last_updated_at TIMESTAMP,
            status TEXT,
            logs JSONB,
            user_id UUID REFERENCES users(user_id)
        );
        CREATE INDEX idx_integration_data_integration_type ON integration_data(integration_type);
        CREATE INDEX idx_integration_data_status ON integration_data(status);
        CREATE INDEX idx_integration_data_user_id ON integration_data(user_id);
        """,
        """
        CREATE TABLE reports (
            report_id UUID PRIMARY KEY,
            created_at TIMESTAMP NOT NULL,
            report_data JSONB,
            user_id UUID REFERENCES users(user_id),
            report_type TEXT
        );
        CREATE INDEX idx_reports_created_at ON reports(created_at);
        CREATE INDEX idx_reports_user_id ON reports(user_id);
        CREATE INDEX idx_reports_report_type ON reports(report_type);
        """,
        """
        CREATE TABLE history (
            history_id UUID PRIMARY KEY,
            table_name TEXT,
            record_id UUID,
            change_timestamp TIMESTAMP NOT NULL,
            change_type TEXT,
            old_values JSONB,
            new_values JSONB,
            user_id UUID REFERENCES users(user_id)
        );
        CREATE INDEX idx_history_table_name ON history(table_name);
        CREATE INDEX idx_history_record_id ON history(record_id);
        CREATE INDEX idx_history_change_timestamp ON history(change_timestamp);
        CREATE INDEX idx_history_change_type ON history(change_type);
        CREATE INDEX idx_history_user_id ON history(user_id);
        """,
        """
        CREATE TABLE knowledgebase (
            entry_id UUID PRIMARY KEY,
            entry_data JSONB NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            last_modified TIMESTAMP NOT NULL,
            version INTEGER,
            user_id UUID REFERENCES users(user_id),
            interaction_id UUID REFERENCES user_interactions(interaction_id),
            data_id UUID,
            taxonomy_id UUID,
            model_metrics_id UUID REFERENCES model_metrics(id),
            session_id UUID,  
            feedback_id UUID,  
            integration_data_id UUID REFERENCES integration_data(id),
            report_id UUID REFERENCES reports(report_id),
            history_id UUID REFERENCES history(history_id),
            embeddings_id UUID REFERENCES embeddings(vector_id),
            FOREIGN KEY (session_id, feedback_id) REFERENCES sessions_feedback(session_id, feedback_id) 
        );
        CREATE INDEX idx_knowledgebase_timestamp ON knowledgebase(timestamp);
        CREATE INDEX idx_knowledgebase_last_modified ON knowledgebase(last_modified);
        """,
        "ALTER TABLE data ADD CONSTRAINT data_knowledgebase_fk FOREIGN KEY (knowledgebase_id) REFERENCES knowledgebase(entry_id);",
        "ALTER TABLE knowledgebase ADD CONSTRAINT knowledgebase_data_fk FOREIGN KEY (data_id) REFERENCES data(id);",
        "ALTER TABLE knowledgebase ADD CONSTRAINT knowledgebase_taxonomy_fk FOREIGN KEY (taxonomy_id) REFERENCES taxonomy(id);"
    ]


    # Execute each command
    for command in commands:
        try:
            cursor.execute(command)
        except Exception as e:
            logger.error(f"Failed to execute command. Error: {str(e)}")
            raise

    # Commit the transaction
    conn.commit()

    logger.info("Schema created successfully.")



def main():
    logger.info("Testing database connection...")
    conn = get_db_connection()

    if conn is not None:
        logger.info("Database connection successful.")

        logger.info("Creating schema...")
        create_schema(conn)

        conn.close()
        logger.info("Database connection closed.")

if __name__ == "__main__":
    main()

