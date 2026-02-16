import sqlite3
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

class SemanticSearchAgent:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.qdrant = QdrantClient(path="./qdrant_db")
        self.collection = "media_questions"

    def retrieve_sql_templates(self, question, top_k=3):
        q_emb = self.embedder.encode([question]).tolist()
        results = self.qdrant.query_points(
            collection_name=self.collection,
            query=q_emb[0],
            limit=top_k,
            with_payload=True
        )
        templates = []
        for point in results.points:
            templates.append({
                "question": point.payload.get("question", ""),
                "sql_template": point.payload.get("sql_template", ""),
                "score": point.score
            })
        return templates

    def generate_sql(self, question, templates):
        if not templates:
            return None
        return templates[0]["sql_template"]

    def execute_sql(self, sql):
        conn = sqlite3.connect('wbd_sample.db')
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return columns, results
        except Exception as e:
            return None, str(e)
        finally:
            conn.close()

    def answer(self, question):
        templates = self.retrieve_sql_templates(question)
        sql = self.generate_sql(question, templates)
        if not sql:
            return {"answer": "Sorry, I couldn't find a relevant template."}
        cols, data = self.execute_sql(sql)
        if data is None:
            return {"answer": f"SQL error: {data}"}
        answer = f"Found {len(data)} results.\n"
        for row in data[:5]:
            answer += str(row) + "\n"
        return {"answer": answer, "sql": sql}
