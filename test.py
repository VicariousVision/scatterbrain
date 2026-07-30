from dotenv import load_dotenv
import os

load_dotenv()  # This must be called first

neo4j_uri = os.getenv('NEO4J_URI')
neo4j_password = os.getenv('NEO4J_PASSWORD')

print(f"URI: {neo4j_uri}")
print(f"Password: {neo4j_password}")  # Debug: verify it's loaded