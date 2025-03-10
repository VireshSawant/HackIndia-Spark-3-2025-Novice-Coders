import os
import json
import time
import base64
import hashlib
import pickle
from datetime import datetime
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud

import nltk, re, requests
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from transformers import pipeline
# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# ----------------------------------
# Helper: Highlight query terms in snippet using HTML <mark> tags.
def highlight_query_terms(text, query):
    # Split query into individual words (lowercase)
    query_terms = query.lower().split()
    # For each term, replace its occurrences in the text with a highlighted version.
    # We use <mark> tag and allow unsafe HTML in st.markdown.
    highlighted_text = text
    for term in query_terms:
        # Use regex for case-insensitive replacement
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        highlighted_text = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", highlighted_text)
    return highlighted_text

# ----------------------------------
# Module: DocumentProcessor
import fitz  # PyMuPDF
import docx  # python-docx

class DocumentProcessor:
    def __init__(self, storage_dir: str = "document_storage"):
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

    def process_document(self, file_path: str) -> dict:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.pdf':
            text, metadata = self._process_pdf(file_path)
        elif file_extension == '.docx':
            text, metadata = self._process_docx(file_path)
        elif file_extension == '.txt':
            text, metadata = self._process_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        doc_id = self._generate_document_id(text)
        chunks = self._chunk_text(text)
        doc_info = {
            "id": doc_id,
            "filename": os.path.basename(file_path),
            "metadata": metadata,
            "text": text,
            "chunks": chunks,
            "created_at": datetime.now().isoformat()
        }
        self._save_processed_document(doc_info)
        return doc_info

    def _process_pdf(self, file_path: str) -> tuple:
        text = ""
        doc = fitz.open(file_path)
        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "pages": len(doc)
        }
        for page in doc:
            text += page.get_text()
        return text, metadata

    def _process_docx(self, file_path: str) -> tuple:
        document = docx.Document(file_path)
        text = "\n".join([para.text for para in document.paragraphs])
        metadata = {
            "title": document.core_properties.title if hasattr(document, 'core_properties') else "",
            "author": document.core_properties.author if hasattr(document, 'core_properties') else "",
            "pages": len(document.sections)
        }
        return text, metadata

    def _process_txt(self, file_path: str) -> tuple:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        metadata = {"title": os.path.basename(file_path), "pages": 1}
        return text, metadata

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                words = current_chunk.split()
                if len(words) > overlap:
                    current_chunk = " ".join(words[-overlap:]) + " " + sentence
                else:
                    current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def _generate_document_id(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    def _save_processed_document(self, doc_info: dict) -> None:
        doc_path = os.path.join(self.storage_dir, f"{doc_info['id']}.json")
        with open(doc_path, 'w', encoding='utf-8') as f:
            json.dump(doc_info, f, ensure_ascii=False, indent=2)

# ----------------------------------
# Module: SemanticSearchEngine
from sentence_transformers import SentenceTransformer

class SemanticSearchEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_dir: str = "vector_index"):
        self.model = SentenceTransformer(model_name)
        self.index_dir = index_dir
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        self.document_index = {}  # {doc_id: {chunk_idx: embedding}}
        self.document_info = {}   # {doc_id: lightweight document metadata}
        self._load_index()
        # Instantiate the QueryRefiner for snippet extraction
        self.query_refiner = QueryRefiner()

    def index_document(self, doc_info: dict) -> None:
        doc_id = doc_info["id"]
        chunks = doc_info["chunks"]
        chunk_embeddings = {}
        for idx, chunk in enumerate(chunks):
            embedding = self.model.encode(chunk)
            chunk_embeddings[idx] = embedding
        self.document_index[doc_id] = chunk_embeddings
        doc_info_light = {
            "id": doc_info["id"],
            "filename": doc_info["filename"],
            "metadata": doc_info["metadata"],
            "chunk_count": len(chunks)
        }
        self.document_info[doc_id] = doc_info_light
        self._save_index()

    def search(self, query: str, top_k: int = 5) -> list:
        query_embedding = self.model.encode(query)
        results = []
        for doc_id, chunk_embeddings in self.document_index.items():
            for chunk_idx, embedding in chunk_embeddings.items():
                similarity = self._cosine_similarity(query_embedding, embedding)
                results.append({
                    "doc_id": doc_id,
                    "chunk_idx": chunk_idx,
                    "similarity": similarity,
                    "doc_info": self.document_info[doc_id]
                })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = results[:top_k]
        # Use the QA model to extract a relevant snippet for each candidate
        for result in top_results:
            doc_path = os.path.join("document_storage", f"{result['doc_id']}.json")
            with open(doc_path, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            context = doc_data["chunks"][result["chunk_idx"]]
            qa_result = self.query_refiner.extract_relevant_snippet(query, context)
            result["qa_answer"] = qa_result.get("answer", "")
            result["qa_score"] = qa_result.get("score", 0)
        # Re-sort by QA score to boost results that have a clear answer snippet.
        top_results.sort(key=lambda x: x["qa_score"], reverse=True)
        return top_results

    def _get_full_results(self, results: list) -> list:
        for result in results:
            doc_path = os.path.join("document_storage", f"{result['doc_id']}.json")
            with open(doc_path, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            result["text"] = doc_data["chunks"][result["chunk_idx"]]
        return results

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _save_index(self) -> None:
        index_path = os.path.join(self.index_dir, "document_index.pkl")
        info_path = os.path.join(self.index_dir, "document_info.json")
        with open(index_path, 'wb') as f:
            pickle.dump(self.document_index, f)
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(self.document_info, f, ensure_ascii=False, indent=2)

    def _load_index(self) -> None:
        index_path = os.path.join(self.index_dir, "document_index.pkl")
        info_path = os.path.join(self.index_dir, "document_info.json")
        if os.path.exists(index_path) and os.path.exists(info_path):
            with open(index_path, 'rb') as f:
                self.document_index = pickle.load(f)
            with open(info_path, 'r', encoding='utf-8') as f:
                self.document_info = json.load(f)

# ----------------------------------
# Module: QueryRefiner (using QA for snippet extraction)
class QueryRefiner:
    def __init__(self, model_name: str = "deepset/roberta-base-squad2"):
        self.qa_pipeline = pipeline("question-answering", model=model_name)
    
    def extract_relevant_snippet(self, query: str, context: str) -> dict:
        try:
            result = self.qa_pipeline(question=query, context=context)
            return result
        except Exception as e:
            print("QA extraction error:", e)
            return {"answer": "", "score": 0}

# ----------------------------------
# Module: QueryProcessor (remains similar)
class QueryProcessor:
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.intent_patterns = {
            'find': r"(?:find|search|locate|get|retrieve|show|give)\s+(?:me\s+)?(?:information|info|documents|docs|files|results)?(?:\s+about|\s+on|\s+regarding)?",
            'summarize': r"(?:summarize|summarise|summary|brief|overview|gist|recap)\s+(?:of|about|on)?",
        }

    def process_query(self, query: str) -> dict:
        cleaned_query = self._clean_query(query)
        intent = self._detect_intent(query)
        key_terms = self._extract_key_terms(cleaned_query)
        expanded_query = self._expand_query(cleaned_query, key_terms)
        return {
            "original_query": query,
            "cleaned_query": cleaned_query,
            "intent": intent,
            "key_terms": key_terms,
            "expanded_query": expanded_query
        }

    def _clean_query(self, query: str) -> str:
        query = query.lower()
        query = re.sub(r'[^\w\s]', ' ', query)
        query = re.sub(r'\s+', ' ', query).strip()
        return query

    def _detect_intent(self, query: str) -> str:
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, query.lower()):
                return intent
        return "search"

    def _extract_key_terms(self, query: str) -> list:
        words = query.split()
        words = [word for word in words if word not in self.stopwords]
        words = [self.lemmatizer.lemmatize(word) for word in words]
        pos_tags = nltk.pos_tag(words)
        key_terms = []
        for word, pos in pos_tags:
            if pos.startswith('NN') or (pos.startswith('VB') and len(word) > 2):
                key_terms.append(word)
        return key_terms if key_terms else words

    def _expand_query(self, query: str, key_terms: list) -> str:
        expanded_terms = []
        for term in key_terms:
            synonyms = self._get_synonyms(term)
            expanded_terms.append(term)
            expanded_terms.extend(synonyms)
        expanded_query = query + " " + " ".join(expanded_terms)
        return expanded_query

    def _get_synonyms(self, term: str) -> list:
        synonym_dict = {
            "document": ["file", "paper", "text", "report"],
            "search": ["find", "retrieve", "locate", "get"],
            "summarize": ["summarise", "recap", "overview", "brief"]
        }
        return synonym_dict.get(term, [])

# ----------------------------------
# Module: DocumentSummarizer (using Hugging Face Inference API)
class DocumentSummarizer:
    def __init__(self, hf_api_token: str, model_name: str = "google/pegasus-xsum"):
        self.hf_api_token = hf_api_token
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {self.hf_api_token}"}
    
    def query(self, payload: dict, retries=3, delay=5) -> dict:
        for attempt in range(retries):
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            if response.status_code == 200:
                try:
                    return response.json()
                except Exception as e:
                    print("JSON decode error:", e)
            elif response.status_code == 503:
                print(f"503 received, retrying in {delay} seconds (attempt {attempt+1}/{retries})...")
                time.sleep(delay)
            else:
                print("Response status:", response.status_code)
                print("Response text:", response.text)
                raise Exception(f"API request failed with status code {response.status_code}")
        raise Exception("Max retries reached. API is unavailable.")
    
    def split_text(self, text: str, max_tokens: int = 512) -> list:
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk.split()) + len(sentence.split()) > max_tokens:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def summarize(self, text: str, max_length: int = 150) -> dict:
        chunks = self.split_text(text, max_tokens=512)
        summaries = []
        for chunk in chunks:
            payload = {
                "inputs": chunk,
                "parameters": {"max_length": max_length, "min_length": 40, "do_sample": False}
            }
            result = self.query(payload)
            if isinstance(result, list) and "summary_text" in result[0]:
                summaries.append(result[0]["summary_text"])
            else:
                summaries.append("Summarization Error.")
        final_summary = " ".join(summaries)
        return {"summary": final_summary, "method": "HuggingFace Inference API"}

# ----------------------------------
# Module: UserPersonalization
class UserPersonalization:
    def __init__(self, user_data_dir: str = "user_data"):
        self.user_data_dir = user_data_dir
        if not os.path.exists(user_data_dir):
            os.makedirs(user_data_dir)

    def get_user_profile(self, user_id: str) -> dict:
        profile_path = os.path.join(self.user_data_dir, f"{user_id}_profile.json")
        if os.path.exists(profile_path):
            with open(profile_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            new_profile = {
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "preferences": {
                    "summary_length": "medium",
                    "result_count": 5,
                    "highlight_color": "#FFFF00",
                    "theme": "light"
                },
                "interests": [],
                "search_history": [],
                "document_history": {},
                "topic_affinity": {}
            }
            self._save_user_profile(user_id, new_profile)
            return new_profile

    def update_preferences(self, user_id: str, preferences: dict) -> dict:
        profile = self.get_user_profile(user_id)
        profile["preferences"].update(preferences)
        self._save_user_profile(user_id, profile)
        return profile["preferences"]

    def log_search(self, user_id: str, query: str, results: list) -> None:
        profile = self.get_user_profile(user_id)
        search_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "result_count": len(results),
            "result_ids": [r.get("doc_id") for r in results[:5]]
        }
        profile["search_history"].append(search_entry)
        if len(profile["search_history"]) > 100:
            profile["search_history"] = profile["search_history"][-100:]
        self._save_user_profile(user_id, profile)

    def log_document_view(self, user_id: str, doc_info: dict) -> None:
        profile = self.get_user_profile(user_id)
        profile["document_history"][doc_info.get("id")] = {
            "timestamp": datetime.now().isoformat(),
            "filename": doc_info.get("filename"),
            "metadata": doc_info.get("metadata", {})
        }
        self._save_user_profile(user_id, profile)

    def _save_user_profile(self, user_id: str, profile: dict) -> None:
        profile_path = os.path.join(self.user_data_dir, f"{user_id}_profile.json")
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

# ----------------------------------
# Module: DocumentOrganizer
class DocumentOrganizer:
    def __init__(self, storage_dir: str = "document_organization"):
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        self.stopwords = set(stopwords.words('english'))
        self.predefined_categories = [
            "business", "technology", "science", "health", "finance",
            "legal", "education", "marketing", "human resources", "research"
        ]
        self.organization_data = self._load_organization_data()

    def classify_document(self, doc_info: dict) -> dict:
        doc_id = doc_info["id"]
        text = doc_info["text"]
        keywords = self._extract_keywords(text)
        categories = self._assign_categories(text, keywords)
        org_entry = {
            "doc_id": doc_id,
            "keywords": keywords,
            "categories": categories,
            "filename": doc_info.get("filename", ""),
            "metadata": doc_info.get("metadata", {})
        }
        self.organization_data[doc_id] = org_entry
        self._save_organization_data()
        return org_entry

    def organize_documents(self, doc_list: list, method: str = "cluster") -> dict:
        for doc in doc_list:
            if doc["id"] not in self.organization_data:
                self.classify_document(doc)
        if method == "cluster":
            return self._cluster_documents(doc_list)
        elif method == "topic":
            return self._topic_modeling(doc_list)
        else:
            return self._category_organization(doc_list)

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> list:
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in self.stopwords and len(word) > 3]
        word_freq = Counter(words)
        keywords = [word for word, count in word_freq.most_common(max_keywords * 2)]
        generic_terms = ["would", "could", "should", "might", "about", "many", "some", "have", "this", "that", "these", "those"]
        keywords = [kw for kw in keywords if kw not in generic_terms]
        return keywords[:max_keywords]

    def _assign_categories(self, text: str, keywords: list) -> list:
        text_lower = text.lower()
        assigned_categories = []
        category_keywords = {
            "business": ["business", "company", "market", "industry", "corporate", "management", "strategy", "profit"],
            "technology": ["technology", "software", "hardware", "digital", "computer", "system", "data", "programming"],
            "science": ["science", "research", "experiment", "theory", "study", "analysis"],
            "health": ["health", "medical", "medicine", "patient", "doctor", "treatment", "disease", "hospital"],
            "finance": ["finance", "financial", "money", "investment", "fund", "bank", "stock", "budget"],
            "legal": ["legal", "law", "regulation", "compliance", "policy", "contract", "agreement", "rights"],
            "education": ["education", "school", "student", "teacher", "learning", "teaching", "course", "training"],
            "marketing": ["marketing", "brand", "customer", "advertising", "campaign", "sales", "market", "promotion"],
            "human resources": ["hr", "human resources", "employee", "hiring", "recruitment", "personnel", "staff"],
            "research": ["research", "development", "innovation", "discovery", "investigation", "analysis", "findings"]
        }
        for category, cat_keywords in category_keywords.items():
            score = 0
            for kw in cat_keywords:
                if kw in text_lower:
                    score += text_lower.count(kw) * 2
            for kw in keywords:
                if kw in cat_keywords:
                    score += 5
            if score >= 10:
                assigned_categories.append(category)
        if not assigned_categories:
            scores = {category: 0 for category in self.predefined_categories}
            for category, cat_keywords in category_keywords.items():
                for kw in cat_keywords:
                    if kw in text_lower:
                        scores[category] += text_lower.count(kw)
                for kw in keywords:
                    if kw in cat_keywords or any(cat_kw in kw for cat_kw in cat_keywords):
                        scores[category] += 2
            best_category = max(scores, key=scores.get)
            if scores[best_category] > 0:
                assigned_categories.append(best_category)
            else:
                assigned_categories.append("uncategorized")
        return assigned_categories

    def _cluster_documents(self, doc_list: list) -> dict:
        if len(doc_list) < 2:
            return {"clusters": [{"name": "All Documents", "docs": doc_list}]}
        doc_texts = [doc.get("text", "") for doc in doc_list]
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(doc_texts)
        num_clusters = min(max(2, len(doc_list) // 5), 10)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
        feature_names = vectorizer.get_feature_names_out()
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        cluster_docs = {i: [] for i in range(num_clusters)}
        for i, cluster_id in enumerate(clusters):
            cluster_docs[cluster_id].append(doc_list[i])
        result = {"clusters": []}
        for cluster_id, docs in cluster_docs.items():
            top_terms = [feature_names[idx] for idx in order_centroids[cluster_id, :5]]
            cluster_name = ", ".join(top_terms).title()
            result["clusters"].append({
                "name": cluster_name,
                "docs": docs,
                "top_terms": top_terms,
                "size": len(docs)
            })
        result["clusters"].sort(key=lambda x: x["size"], reverse=True)
        return result

    def _topic_modeling(self, doc_list: list) -> dict:
        if len(doc_list) < 3:
            return {"topics": [{"name": "All Documents", "docs": doc_list}]}
        doc_texts = [doc.get("text", "") for doc in doc_list]
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(doc_texts)
        num_topics = min(max(2, len(doc_list) // 8), 8)
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(tfidf_matrix)
        feature_names = vectorizer.get_feature_names_out()
        topic_terms = []
        for topic_idx, topic in enumerate(lda.components_):
            top_terms = [feature_names[i] for i in topic.argsort()[:-11:-1]]
            topic_terms.append(top_terms)
        doc_topic_dist = lda.transform(tfidf_matrix)
        doc_primary_topics = np.argmax(doc_topic_dist, axis=1)
        topic_docs = {i: [] for i in range(num_topics)}
        for i, topic_id in enumerate(doc_primary_topics):
            topic_docs[topic_id].append(doc_list[i])
        result = {"topics": []}
        for topic_id, docs in topic_docs.items():
            topic_name = " & ".join(topic_terms[topic_id][:3]).title()
            result["topics"].append({
                "name": topic_name,
                "docs": docs,
                "terms": topic_terms[topic_id],
                "size": len(docs)
            })
        result["topics"].sort(key=lambda x: x["size"], reverse=True)
        return result

    def _category_organization(self, doc_list: list) -> dict:
        for doc in doc_list:
            if doc["id"] not in self.organization_data:
                self.classify_document(doc)
        categories = {}
        for doc in doc_list:
            doc_id = doc["id"]
            doc_categories = self.organization_data[doc_id]["categories"]
            for category in doc_categories:
                if category not in categories:
                    categories[category] = []
                categories[category].append(doc)
        if not categories or all(len(docs) == 0 for docs in categories.values()):
            return {"categories": [{"name": "Uncategorized", "docs": doc_list}]}
        result = {"categories": []}
        for category, docs in categories.items():
            result["categories"].append({
                "name": category.title(),
                "docs": docs,
                "size": len(docs)
            })
        result["categories"].sort(key=lambda x: x["size"], reverse=True)
        return result

    def _load_organization_data(self) -> dict:
        data_path = os.path.join(self.storage_dir, "organization_data.json")
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_organization_data(self) -> None:
        data_path = os.path.join(self.storage_dir, "organization_data.json")
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(self.organization_data, f, ensure_ascii=False, indent=2)

# ----------------------------------
# Module: DocumentSearchDashboard (Streamlit App)
class DocumentSearchDashboard:
    def __init__(self, doc_processor, search_engine, query_processor, summarizer, user_personalization, document_organizer, documents):
        st.set_page_config(page_title="AI Document Search Assistant", page_icon="📚", layout="wide")
        self.doc_processor = doc_processor
        self.search_engine = search_engine
        self.query_processor = query_processor
        self.summarizer = summarizer
        self.user_personalization = user_personalization
        self.document_organizer = document_organizer
        self.documents = documents  # List of all processed document info

        # Use URL-safe base64 for user ID to avoid problematic characters
        if 'user_id' not in st.session_state:
            st.session_state['user_id'] = "user_" + base64.urlsafe_b64encode(os.urandom(6)).decode('ascii')
        if 'current_page' not in st.session_state:
            st.session_state['current_page'] = "search"
        if 'search_results' not in st.session_state:
            st.session_state['search_results'] = []
        if 'selected_doc' not in st.session_state:
            st.session_state['selected_doc'] = None
        if 'document_view_mode' not in st.session_state:
            st.session_state['document_view_mode'] = "text"
        # Save the last query for highlighting purposes
        if 'last_query' not in st.session_state:
            st.session_state['last_query'] = ""

    def run(self):
        self._create_sidebar()
        current_page = st.session_state['current_page']
        if current_page == "search":
            self._render_search_page()
        elif current_page == "documents":
            self._render_documents_page()
        elif current_page == "upload":
            self._render_upload_page()
        elif current_page == "profile":
            self._render_profile_page()
        elif current_page == "document_view":
            self._render_document_view()
        elif current_page == "explore":
            self._render_explore_page()

    def _create_sidebar(self):
        with st.sidebar:
            st.title("📚 AI Document Assistant")
            st.subheader("Navigation")
            if st.sidebar.button("🔍 Search", use_container_width=True):
                st.session_state['current_page'] = "search"
            if st.sidebar.button("📂 My Documents", use_container_width=True):
                st.session_state['current_page'] = "documents"
            if st.sidebar.button("⬆️ Upload Documents", use_container_width=True):
                st.session_state['current_page'] = "upload"
            if st.sidebar.button("🧭 Explore Collection", use_container_width=True):
                st.session_state['current_page'] = "explore"
            if st.sidebar.button("👤 My Profile", use_container_width=True):
                st.session_state['current_page'] = "profile"
            st.sidebar.divider()
            st.sidebar.text(f"User: {st.session_state['user_id']}")
            st.sidebar.divider()
            st.sidebar.subheader("Collection Stats")
            st.sidebar.metric("Documents", f"{len(self.documents)}")
            st.sidebar.metric("Pages Indexed", "N/A")
            st.sidebar.metric("Categories", "N/A")

    def _render_search_page(self):
        st.title("🔍 Intelligent Document Search")
        with st.form("search_form"):
            col1, col2 = st.columns([4, 1])
            with col1:
                query = st.text_input("Enter your search query:", placeholder="What are the key financial risks?")
            with col2:
                search_type = st.selectbox("Search Type", ["Semantic", "Keyword", "Combined"])
            submitted = st.form_submit_button("Search", use_container_width=True)
            if submitted and query:
                st.session_state["last_query"] = query  # store query for highlighting
                results = self.search_engine.search(query)
                st.session_state['search_results'] = results
                self.user_personalization.log_search(st.session_state['user_id'], query, results)
        if st.session_state['search_results']:
            st.subheader(f"Search Results ({len(st.session_state['search_results'])} matches)")
            for i, result in enumerate(st.session_state['search_results']):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        title = result['doc_info'].get("filename", "Untitled Document")
                        st.markdown(f"### {title}")
                        snippet = result.get("qa_answer", result.get("text", ""))
                        # Highlight query terms in the snippet
                        highlighted_snippet = highlight_query_terms(snippet, st.session_state["last_query"])
                        st.markdown(highlighted_snippet, unsafe_allow_html=True)
                    with col2:
                        st.metric("Relevance", f"{result['qa_score']*100:.1f}%")
                        if st.button("View Document", key=f"view_{i}"):
                            doc_path = os.path.join("document_storage", f"{result['doc_id']}.json")
                            with open(doc_path, 'r', encoding='utf-8') as f:
                                st.session_state['selected_doc'] = json.load(f)
                            st.session_state['current_page'] = "document_view"
                st.divider()

    def _render_documents_page(self):
        st.title("📂 My Documents")
        docs = self.documents
        if not docs:
            st.info("No documents found. Please upload documents.")
            return
        df = pd.DataFrame([{
            "ID": d["id"],
            "Title": d["filename"],
            "Uploaded": d.get("created_at", "N/A")
        } for d in docs])
        st.dataframe(df, use_container_width=True)
        st.button("Refresh", on_click=self._refresh_documents)

    def _refresh_documents(self):
        docs = []
        storage_dir = "document_storage"
        if os.path.exists(storage_dir):
            for filename in os.listdir(storage_dir):
                if filename.endswith(".json"):
                    with open(os.path.join(storage_dir, filename), "r", encoding="utf-8") as f:
                        docs.append(json.load(f))
        self.documents = docs

    def _render_upload_page(self):
        st.title("⬆️ Upload Documents")
        with st.form("upload_form"):
            uploaded_files = st.file_uploader("Choose files to upload", type=["pdf", "docx", "txt"], accept_multiple_files=True)
            categories = st.multiselect("Categories", ["Business", "Finance", "Technology", "Legal", "Research", "Marketing", "HR", "Personal"])
            access = st.radio("Access Level", ["Private", "Shared", "Public"])
            submitted = st.form_submit_button("Upload and Process", use_container_width=True)
            if submitted and uploaded_files:
                progress_bar = st.progress(0)
                count = len(uploaded_files)
                for i, file in enumerate(uploaded_files):
                    temp_dir = "temp"
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir)
                    temp_path = os.path.join(temp_dir, file.name)
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())
                    doc_info = self.doc_processor.process_document(temp_path)
                    self.search_engine.index_document(doc_info)
                    self.documents.append(doc_info)
                    progress_bar.progress((i+1)/count)
                    os.remove(temp_path)
                st.success(f"Successfully uploaded and processed {count} documents!")
                self._refresh_documents()

    def _render_profile_page(self):
        st.title("👤 User Profile")
        tab1, tab2 = st.tabs(["Preferences", "Search History"])
        user_id = st.session_state['user_id']
        profile = self.user_personalization.get_user_profile(user_id)
        with tab1:
            with st.form("preferences_form"):
                col1, col2 = st.columns(2)
                with col1:
                    theme = st.selectbox("Theme", ["Light", "Dark", "System Default"], index=0)
                    result_count = st.slider("Results per page", 5, 50, profile["preferences"].get("result_count", 5))
                with col2:
                    highlight_color = st.color_picker("Highlight Color", profile["preferences"].get("highlight_color", "#FFFF00"))
                    summary_length = st.select_slider("Summary Length", ["Short", "Medium", "Long"], value=profile["preferences"].get("summary_length", "Medium"))
                submitted = st.form_submit_button("Save Preferences", use_container_width=True)
                if submitted:
                    new_prefs = {"theme": theme, "result_count": result_count, "highlight_color": highlight_color, "summary_length": summary_length}
                    self.user_personalization.update_preferences(user_id, new_prefs)
                    st.success("Preferences updated!")
        with tab2:
            history = profile.get("search_history", [])
            if history:
                df = pd.DataFrame(history)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No search history available.")

    def _render_document_view(self):
        if not st.session_state.get('selected_doc'):
            st.warning("No document selected")
            st.button("Return to Search", on_click=lambda: st.session_state.update({"current_page": "search"}))
            return

        doc = st.session_state['selected_doc']
        st.title(doc.get('filename', 'Untitled Document'))
        st.caption(f"Uploaded: {doc.get('created_at', 'N/A')}")
        
        if st.session_state.get('document_view_mode') not in ["text", "summary", "visual"]:
            st.session_state['document_view_mode'] = "text"
        
        view_options = ["text", "summary", "visual"]
        view_mode = st.radio("View Mode", view_options, horizontal=True, index=view_options.index(st.session_state['document_view_mode']))
        st.session_state['document_view_mode'] = view_mode

        if view_mode == "text":
            st.text_area("Document Text", doc.get('text', 'No text available.'), height=300)
        elif view_mode == "summary":
            summary_result = self.summarizer.summarize(doc.get('text', ''))
            st.markdown("### Document Summary")
            st.markdown(summary_result.get("summary", ""))
        elif view_mode == "visual":
            st.markdown("### Document Visualization")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Word Cloud")
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(doc.get('text', ''))
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            with col2:
                st.subheader("Topic Distribution")
                topics = {"Topic A": 40, "Topic B": 30, "Topic C": 20, "Topic D": 10}
                fig, ax = plt.subplots()
                ax.pie(topics.values(), labels=topics.keys(), autopct='%1.1f%%')
                st.pyplot(fig)
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button("⬅️ Back to Results", on_click=lambda: st.session_state.update({"current_page": "search"}))
        with col2:
            doc_text = doc.get('text', '')
            doc_filename = doc.get('filename', 'document.txt')
            st.download_button(label="📥 Download", data=doc_text, file_name=doc_filename, mime='text/plain')
        with col3:
            st.button("📝 Add Notes", use_container_width=True)
        with col4:
            st.button("🔗 Share Document", use_container_width=True)
        self.user_personalization.log_document_view(st.session_state['user_id'], doc)

    def _render_explore_page(self):
        st.title("🧭 Explore Document Collection")
        tab1, tab2, tab3 = st.tabs(["Category Distribution", "Timeline View", "Topic Network"])
        with tab1:
            st.subheader("Documents by Category")
            org = self.document_organizer._category_organization(self.documents)
            categories = {entry["name"]: entry["size"] for entry in org.get("categories", [])}
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(categories.keys(), categories.values(), color='skyblue')
            ax.set_ylabel("Number of Documents")
            ax.set_title("Document Distribution by Category")
            for i, v in enumerate(categories.values()):
                ax.text(i, v + 0.5, str(v), ha='center')
            st.pyplot(fig)
        with tab2:
            st.subheader("Document Timeline")
            timeline_data = {}
            for doc in self.documents:
                date = doc.get("created_at", "")[:7]
                timeline_data[date] = timeline_data.get(date, 0) + 1
            fig, ax = plt.subplots(figsize=(12, 6))
            dates = list(timeline_data.keys())
            counts = list(timeline_data.values())
            ax.plot(dates, counts, marker='o', linestyle='-', linewidth=2, markersize=8)
            ax.set_xlabel("Month")
            ax.set_ylabel("Documents Added")
            ax.set_title("Document Addition Timeline")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
        with tab3:
            st.subheader("Document Topic Network")
            G = nx.Graph()
            main_topics = ["Finance", "Business", "Technology", "Legal", "Research"]
            G.add_nodes_from(main_topics)
            subtopics = {
                "Finance": ["Risk Management", "Investment", "Accounting", "Taxation"],
                "Business": ["Strategy", "Operations", "Marketing", "HR"],
                "Technology": ["Software", "Hardware", "Cloud", "Security"],
                "Legal": ["Compliance", "Contracts", "IP", "Regulations"],
                "Research": ["Market Analysis", "Product Development", "Competition", "Trends"]
            }
            all_subtopics = []
            for topic_list in subtopics.values():
                all_subtopics.extend(topic_list)
            G.add_nodes_from(all_subtopics)
            for main_topic, topic_list in subtopics.items():
                for subtopic in topic_list:
                    G.add_edge(main_topic, subtopic)
            cross_connections = [
                ("Risk Management", "Compliance"),
                ("Investment", "Strategy"),
                ("Cloud", "Security"),
                ("Product Development", "Marketing"),
                ("Taxation", "Regulations"),
                ("Strategy", "Market Analysis"),
                ("HR", "Operations"),
                ("Software", "IP")
            ]
            G.add_edges_from(cross_connections)
            color_map = ['lightblue' if node in main_topics else 'lightgreen' for node in G.nodes()]
            fig, ax = plt.subplots(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=42, k=0.3)
            nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=1500)
            nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6)
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
            ax.axis('off')
            st.pyplot(fig)

# ----------------------------------
# Main entry point
if __name__ == "__main__":
    # Instantiate components
    doc_processor = DocumentProcessor()
    search_engine = SemanticSearchEngine()
    query_processor = QueryProcessor()
    # Use your Hugging Face API token (update if needed)
    summarizer = DocumentSummarizer(hf_api_token="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", model_name="google/pegasus-xsum")
    user_personalization = UserPersonalization()
    document_organizer = DocumentOrganizer()

    # Load all documents from storage
    documents = []
    storage_dir = "document_storage"
    if os.path.exists(storage_dir):
        for filename in os.listdir(storage_dir):
            if filename.endswith(".json"):
                with open(os.path.join(storage_dir, filename), "r", encoding='utf-8') as f:
                    doc_info = json.load(f)
                    documents.append(doc_info)
                    if doc_info["id"] not in search_engine.document_index:
                        search_engine.index_document(doc_info)
    dashboard = DocumentSearchDashboard(doc_processor, search_engine, query_processor, summarizer, user_personalization, document_organizer, documents)
    dashboard.run()
