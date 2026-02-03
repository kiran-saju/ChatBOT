# backend/main.py (Simplified - SQLite Only)
import json
import os
import numpy as np
import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
# Add this AFTER creating your FastAPI app
# Example: after `app = FastAPI()`

# Serve static files (CSS, JS will be available at /static/)
app.mount("/static", StaticFiles(directory="."), name="static")

# Serve index.html at the root URL
@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

# Optional: Also serve other frontend routes for SPA
@app.get("/{path:path}")
async def catch_all(path: str):
    # If it's a file that exists (like favicon.ico), serve it
    if os.path.exists(path) and os.path.isfile(path):
        return FileResponse(path)
    # Otherwise, serve index.html (for React/Vue/Angular routing)
    return FileResponse("index.html")

# ---------- Database Configuration ----------
SQLITE_DB_PATH = "chatbot_fallback.db"

def setup_database():
    """Ensure database tables exist and are populated"""
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cur = conn.cursor()
        
        # Create FAQ table if not exists
        cur.execute('''
            CREATE TABLE IF NOT EXISTS faq(
                faq_id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL UNIQUE,
                answer_text TEXT NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert sample data if empty
        cur.execute("SELECT COUNT(*) FROM faq")
        if cur.fetchone()[0] == 0:
            sample_faqs = [
                ("Free learning courses", "We offer a set of free, self-paced courses. Start anytime."),
                ("Connect to Counsellor", "Our counsellor can guide you on course selection, fees, and eligibility."),
                ("Are You Eligible? Find Out!", "Share your highest qualification and years of experience for a quick check."),
                ("Submit your Application", "Apply online with your ID, CV and transcripts ready.")
            ]
            cur.executemany(
                "INSERT INTO faq (question, answer_text) VALUES (?, ?)",
                sample_faqs
            )
            print("Sample data inserted into database")
        
        conn.commit()
        conn.close()
        print("Database setup completed")
    except Exception as e:
        print(f"Database setup error: {e}")

def get_db_conn():
    """Get SQLite database connection"""
    return sqlite3.connect(SQLITE_DB_PATH)

def execute_query(sql, params=None):
    """Execute query on SQLite database"""
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        
        if sql.strip().upper().startswith('SELECT'):
            result = cur.fetchall()
            conn.close()
            return result
        else:
            conn.commit()
            conn.close()
            return cur.rowcount
    except Exception as e:
        raise e

def get_db_faqs():
    """Get all FAQs from SQLite database"""
    try:
        print("Loading FAQs from SQLite database...")
        
        sql = "SELECT question, answer_text FROM faq WHERE is_active=1"
        rows = execute_query(sql)
        
        faqs = []
        for row in rows:
            faqs.append({
                "question": row[0],
                "answer": row[1],
                "category": "database_faq",
                "keywords": [],
                "confidence": 0.95,
                "source": "database"
            })
        print(f"Successfully loaded {len(faqs)} FAQs from database")
        return faqs
        
    except Exception as e:
        print(f"Error fetching FAQs from database: {e}")
        import traceback
        print(f"Full error traceback: {traceback.format_exc()}")
        return []

class HybridKnowledgeManager:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print("Initializing Hybrid Knowledge Manager...")
        self.model = SentenceTransformer(model_name)
        self.knowledge_base = []
        self.embeddings = None
        self.training_texts = []
        
        # Load all knowledge sources
        self.load_all_knowledge_sources()
    
    def load_all_knowledge_sources(self):
        """Load knowledge from all available sources"""
        self.knowledge_base = []
        
        print("Loading knowledge from all sources...")
        
        # Debug: Show current working directory and files
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in current dir: {os.listdir('.')}")
        if os.path.exists('backend'):
            print(f"Files in backend: {os.listdir('backend')}")
        
        # 1. Load FAQs from Database
        db_faqs = get_db_faqs()
        self.knowledge_base.extend(db_faqs)
        print(f"Loaded {len(db_faqs)} FAQs from database")
        
        # 2. Load JSON knowledge bases
        knowledge_files = {
            'knowledge_base.json': 'json_faq',
            'uniathena_knowledge_base.json': 'company_info', 
            'faq_knowledge_base.json': 'detailed_faq',
            'MEM.json': 'academic_programs'
        }
        
        for file_name, source_name in knowledge_files.items():
            file_path = file_name
            # Try both current directory and backend directory
            if not os.path.exists(file_path) and os.path.exists(f'backend/{file_name}'):
                file_path = f'backend/{file_name}'
                
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        kb_data = json.load(f)
                    self._add_knowledge_from_dict(kb_data, source_name)
                    print(f"Loaded {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            else:
                print(f"{file_path} not found - skipping")
        
        print(f"Total knowledge items loaded: {len(self.knowledge_base)}")
        self.train_model()

    def _add_knowledge_from_dict(self, kb_dict: dict, source: str):
        """Add knowledge from dictionary structure to flat list"""
        for category, items in kb_dict.items():
            for item in items:
                # Ensure all required fields are present
                item['category'] = category
                item['source'] = source
                if 'keywords' not in item:
                    item['keywords'] = []
                if 'confidence' not in item:
                    item['confidence'] = 0.9
                self.knowledge_base.append(item)
    
    def train_model(self):
        """Train the AI model with the complete knowledge base"""
        if not self.knowledge_base:
            print("No knowledge base available for training")
            return
        
        # Create training data from questions, answers, and keywords
        self.training_texts = []
        for item in self.knowledge_base:
            self.training_texts.append(item['question'])
            self.training_texts.append(item['answer'])
            self.training_texts.extend(item.get('keywords', []))
        
        # Remove duplicates
        self.training_texts = list(set(self.training_texts))
        
        if self.training_texts:
            self.embeddings = self.model.encode(self.training_texts)
            print(f"Model trained with {len(self.training_texts)} training texts")
        else:
            print("No training texts available")
    
    def find_best_match(self, user_question: str) -> Dict[str, Any]:
        """Find the best matching answer using semantic search"""
        if not self.knowledge_base or self.embeddings is None:
            return self.get_fallback_response(user_question)
        
        # First, try exact match with database FAQs (your existing system)
        exact_match = self._find_exact_match(user_question)
        if exact_match:
            return exact_match
        
        # If no exact match, use AI semantic search
        try:
            user_embedding = self.model.encode([user_question])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(user_embedding, self.embeddings)[0]
            
            # Find best match
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            
            # Map back to knowledge base item
            training_text_idx = best_match_idx
            best_training_text = self.training_texts[training_text_idx]
            
            # Find which knowledge base item this training text came from
            best_item = None
            for item in self.knowledge_base:
                if (best_training_text == item['question'] or 
                    best_training_text == item['answer'] or 
                    best_training_text in item.get('keywords', [])):
                    best_item = item
                    break
            
            if best_item and best_similarity > 0.6:  # Good match threshold
                return {
                    "answer": best_item['answer'],
                    "confidence": float(best_similarity),
                    "category": best_item['category'],
                    "source": best_item.get('source', 'ai_knowledge_base')
                }
            else:
                return self.get_fallback_response(user_question)
                
        except Exception as e:
            print(f"Error in AI matching: {e}")
            return self.get_fallback_response(user_question)
    
    def _find_exact_match(self, user_question: str) -> Optional[Dict[str, Any]]:
        """Try to find exact match in database FAQs (your existing system)"""
        user_question_clean = user_question.lower().strip()
        
        print(f"Looking for exact match for: '{user_question}'")
        
        for item in self.knowledge_base:
            if item['source'] == 'database':
                # Clean up the question for comparison
                db_question_clean = item['question'].lower().strip()
                
                # Exact match with database questions
                if user_question_clean == db_question_clean:
                    print(f"Exact match found: '{user_question}' = '{item['question']}'")
                    return {
                        "answer": item['answer'],
                        "confidence": 1.0,
                        "category": item['category'],
                        "source": "database_exact_match"
                    }
        
        print("No exact match found")
        return None
    
    def get_predefined_options(self) -> List[Dict[str, str]]:
        """Get predefined options from database for the chat interface"""
        predefined = []
        try:
            sql = "SELECT question FROM faq WHERE is_active=1"
            rows = execute_query(sql)
            for row in rows:
                predefined.append({
                    "text": row[0],
                    "type": "predefined"
                })
        except Exception as e:
            print(f"Error fetching predefined options: {e}")
        
        # Add some AI-powered suggested questions from JSON knowledge
        ai_suggestions = []
        for item in self.knowledge_base:
            if item['source'] != 'database' and len(ai_suggestions) < 4:
                ai_suggestions.append({
                    "text": item['question'],
                    "type": "ai_suggestion"
                })
        
        return predefined + ai_suggestions
    
    def get_fallback_response(self, user_question: str) -> Dict[str, Any]:
        """Provide intelligent fallback responses"""
        user_lower = user_question.lower()
        
        # Enhanced keyword matching
        keyword_responses = {
            'fee': "Course fees range from $500 for certificates to $5000 for comprehensive programs. We offer flexible payment plans.",
            'cost': "Program costs vary based on duration and credits. Please check our website for detailed pricing.",
            'admission': "Admission requires a bachelor's degree or equivalent experience. We have regular intakes throughout the year.",
            'scholarship': "We offer merit-based and need-based scholarships. Contact our admissions team for eligibility details.",
            'course': "We offer programs in Data Science, Business, IT, and more. All courses are fully online and accredited.",
            'duration': "Programs range from 6-24 months depending on the plan. We offer both Fast Track and Flexible options.",
        }
        
        for keyword, response in keyword_responses.items():
            if keyword in user_lower:
                return {
                    "answer": response,
                    "confidence": 0.5,
                    "category": "keyword_match",
                    "source": "fallback"
                }
        
        # General fallback
        import random
        general_responses = [
            f"I understand you're asking about '{user_question}'. For detailed information, please visit our website or contact our support team.",
            f"That's a great question about '{user_question}'. Our support team can provide you with specific details.",
        ]
        
        return {
            "answer": random.choice(general_responses),
            "confidence": 0.2,
            "category": "general",
            "source": "fallback"
        }
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        sources = {}
        categories = {}
        for item in self.knowledge_base:
            src = item.get('source', 'unknown')
            cat = item['category']
            sources[src] = sources.get(src, 0) + 1
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_items": len(self.knowledge_base),
            "sources": sources,
            "categories": categories,
            "training_texts": len(self.training_texts),
            "model_ready": self.embeddings is not None,
            "database_connected": True
        }

# Initialize database first
print("Setting up database...")
setup_database()

# Initialize the hybrid knowledge manager
print("Initializing Hybrid Chatbot System...")
knowledge_manager = HybridKnowledgeManager()

# ---------- FastAPI Setup ----------
app = FastAPI(
    title="UniAthena Hybrid Chatbot API",
    description="Combines existing DB FAQs with AI-powered knowledge base",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic Models ----------
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    confidence: float
    category: str
    source: str

class KnowledgeStats(BaseModel):
    total_items: int
    sources: Dict[str, int]
    categories: Dict[str, int]
    training_texts: int
    model_ready: bool
    database_connected: bool

class PredefinedOptionsResponse(BaseModel):
    options: List[Dict[str, str]]

# ---------- API Endpoints ----------
@app.get("/")
async def root():
    return {
        "message": "UniAthena Hybrid Chatbot API",
        "status": "operational",
        "system": "DB FAQs + AI Knowledge Base",
        "knowledge_base": knowledge_manager.get_knowledge_stats()
    }

@app.get("/health")
async def health():
    stats = knowledge_manager.get_knowledge_stats()
    return {
        "status": "healthy" if stats["model_ready"] else "degraded",
        "database": "sqlite",
        "knowledge_base": stats
    }

@app.get("/stats")
async def get_stats():
    return knowledge_manager.get_knowledge_stats()

@app.get("/options")
async def get_predefined_options():
    """Get predefined options for chat interface"""
    options = knowledge_manager.get_predefined_options()
    return PredefinedOptionsResponse(options=options)

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """Main chatbot endpoint - combines DB and AI responses"""
    try:
        user_query = request.query.strip()
        
        if not user_query:
            return ChatResponse(
                response="Please enter your question.",
                confidence=0.0,
                category="error",
                source="system"
            )
        
        print(f"User query: {user_query}")
        
        # Get hybrid response (DB exact match + AI semantic search)
        response = knowledge_manager.find_best_match(user_query)
        
        print(f"Response - Source: {response['source']}, Confidence: {response['confidence']:.2f}")
        
        return ChatResponse(
            response=response["answer"],
            confidence=response["confidence"],
            category=response["category"],
            source=response["source"]
        )
        
    except Exception as e:
        print(f"Chat error: {e}")
        return ChatResponse(
            response="I'm experiencing technical difficulties. Please try again later.",
            confidence=0.0,
            category="error",
            source="system"
        )

# ---------- Your Existing DB Endpoints (Preserved) ----------
@app.get("/faqs")
async def get_all_faqs():
    """Your existing endpoint to get all FAQs from database"""
    try:
        sql = "SELECT faq_id, question, answer_text, is_active FROM faq WHERE is_active=1"
        rows = execute_query(sql)
        faqs = []
        for row in rows:
            faqs.append({
                "faq_id": row[0],
                "question": row[1],
                "answer_text": row[2],
                "is_active": bool(row[3])
            })
        return faqs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# ---------- Debug Endpoints ----------
@app.get("/test-db")
async def test_database():
    """Test endpoint to check database status"""
    try:
        # Test if FAQ table exists and has data
        sql = "SELECT COUNT(*) FROM faq"
        result = execute_query(sql)
        count = result[0][0] if result else 0
        
        # Also list all tables for SQLite
        tables = []
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        conn.close()
        
        return {
            "database_type": "SQLite",
            "faq_count": count,
            "tables": tables,
            "faq_table_exists": "faq" in tables
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug-knowledge")
async def debug_knowledge():
    """Debug endpoint to see what's in the knowledge base"""
    knowledge_items = []
    
    for i, item in enumerate(knowledge_manager.knowledge_base):
        knowledge_items.append({
            "index": i,
            "question": item.get('question', 'N/A'),
            "answer": item.get('answer', 'N/A')[:100] + "..." if item.get('answer') and len(item.get('answer')) > 100 else item.get('answer', 'N/A'),
            "category": item.get('category', 'N/A'),
            "source": item.get('source', 'N/A'),
            "confidence": item.get('confidence', 'N/A')
        })
    
    return {
        "total_items": len(knowledge_manager.knowledge_base),
        "sources": knowledge_manager.get_knowledge_stats()["sources"],
        "items": knowledge_items
    }

@app.get("/debug-db-faqs")
async def debug_db_faqs():
    """See exactly what's in the FAQ table"""
    try:
        sql = "SELECT question, answer_text FROM faq WHERE is_active=1"
        rows = execute_query(sql)
        faqs = []
        for i, row in enumerate(rows):
            faqs.append({
                "id": i,
                "question": row[0],
                "answer": row[1]
            })
        return {
            "count": len(faqs),
            "faqs": faqs
        }
    except Exception as e:
        return {"error": str(e)}

# ---------- Startup Event ----------
@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print("UniAthena Hybrid Chatbot Started Successfully!")
    print("=" * 60)
    stats = knowledge_manager.get_knowledge_stats()
    print(f"Knowledge Sources:")
    for source, count in stats['sources'].items():
        print(f"   - {source}: {count} items")
    print(f"AI Model: {'Ready' if stats['model_ready'] else 'Not Ready'}")
    print(f"Database: SQLite")
    print(f"API: http://127.0.0.1:8000")
    print(f"Docs: http://127.0.0.1:8000/docs")
    print("=" * 60)

# ---------- Main Execution ----------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
