import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class RAGService:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
        
        self.chunks = []
        self.embeddings = []
        self.chunk_metadata = []
        
    def chunk_document(self, text: str, chunk_size: int = 100, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Split document into overlapping chunks with metadata
        Optimized for CV documents which are typically shorter and more structured
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunk_info = {
                "id": len(chunks),
                "text": chunk_text,
                "start_word": i,
                "end_word": min(i + chunk_size, len(words)),
                "word_count": len(chunk_words),
                "start_char": text.find(chunk_text),
                "end_char": text.find(chunk_text) + len(chunk_text)
            }
            chunks.append(chunk_info)
            
        self.chunks = chunks
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Generate embeddings for each chunk using OpenAI API
        """
        if not self.openai_client:
            # Fallback to TF-IDF if no API key
            return self._generate_tfidf_embeddings(chunks)
        
        embeddings = []
        for chunk in chunks:
            try:
                response = self.openai_client.embeddings.create(
                    input=chunk["text"],
                    model="text-embedding-ada-002"
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error generating embedding: {e}")
                # Fallback to TF-IDF
                return self._generate_tfidf_embeddings(chunks)
        
        self.embeddings = embeddings
        return embeddings
    
    def _generate_tfidf_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Fallback to TF-IDF embeddings if OpenAI API is not available
        """
        texts = [chunk["text"] for chunk in chunks]
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        embeddings = tfidf_matrix.toarray().tolist()
        self.embeddings = embeddings
        return embeddings
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant chunks based on query similarity
        """
        if not self.embeddings:
            return []
        
        # Generate query embedding
        if self.openai_client:
            try:
                response = self.openai_client.embeddings.create(
                    input=query,
                    model="text-embedding-ada-002"
                )
                query_embedding = response.data[0].embedding
            except:
                query_embedding = self._generate_tfidf_embeddings([{"text": query}])[0]
        else:
            query_embedding = self._generate_tfidf_embeddings([{"text": query}])[0]
        
        # Calculate similarities
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities.append((i, similarity))
        
        # Sort by similarity and get top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_chunks = []
        
        for chunk_idx, similarity in similarities[:top_k]:
            chunk_info = self.chunks[chunk_idx].copy()
            chunk_info["similarity_score"] = float(similarity)
            chunk_info["rank"] = len(top_chunks) + 1
            top_chunks.append(chunk_info)
        
        return top_chunks
    
    def generate_rag_response(self, query: str, retrieved_chunks: List[Dict[str, Any]], 
                            cv_text: str) -> Dict[str, Any]:
        """
        Generate RAG response using LLM with retrieved context
        """
        if not retrieved_chunks:
            return {"error": "No relevant chunks found"}
        
        # Prepare context from retrieved chunks
        context = "\n\n".join([f"Chunk {chunk['rank']} (Score: {chunk['similarity_score']:.3f}):\n{chunk['text']}" 
                              for chunk in retrieved_chunks])
        
        # Create prompt for LLM
        prompt = f"""Based on the following CV context and retrieved relevant chunks, answer the question: "{query}"

CV Context:
{cv_text[:1000]}...

Retrieved Relevant Chunks:
{context}

Question: {query}

Please provide a comprehensive answer based on the retrieved information. If the information is not sufficient, indicate what additional information would be needed."""

        try:
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful CV analysis assistant. Provide clear, accurate answers based on the given context."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500
                )
                answer = response.choices[0].message.content
            else:
                # Fallback response
                answer = f"Based on the retrieved chunks, here's what I found:\n\n{context}\n\nQuestion: {query}\n\nNote: This is a fallback response. For better results, please provide an OpenAI API key."
        except Exception as e:
            answer = f"Error generating response: {str(e)}"
        
        return {
            "answer": answer,
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "context_used": context
        }
    
    def get_educational_explanation(self, step: str) -> Dict[str, str]:
        """
        Provide educational explanations for each RAG step
        """
        explanations = {
            "chunking": {
                "title": "Document Chunking",
                "description": "Breaking down the CV into smaller, manageable pieces allows the system to process and analyze specific sections more effectively.",
                "technical_details": "Chunks are created with overlapping text to maintain context between sections. Each chunk is assigned metadata including position, word count, and character boundaries.",
                "benefits": "Enables focused analysis, improves processing efficiency, and allows for targeted retrieval of relevant information."
            },
            "embeddings": {
                "title": "Vector Embeddings",
                "description": "Converting text chunks into numerical vectors (embeddings) that capture semantic meaning and enable similarity calculations.",
                "technical_details": "Using OpenAI's text-embedding-ada-002 model or TF-IDF as fallback. Each chunk becomes a high-dimensional vector representing its semantic content.",
                "benefits": "Enables semantic search, similarity matching, and numerical analysis of text content."
            },
            "retrieval": {
                "title": "Semantic Retrieval",
                "description": "Finding the most relevant chunks based on semantic similarity to the user's query.",
                "technical_details": "Cosine similarity is calculated between the query embedding and all chunk embeddings. Top-k most similar chunks are retrieved and ranked.",
                "benefits": "Provides contextually relevant information, improves answer quality, and enables targeted information extraction."
            },
            "generation": {
                "title": "LLM Response Generation",
                "description": "Using a Large Language Model to synthesize information from retrieved chunks and generate coherent, contextual answers.",
                "technical_details": "The LLM receives the query, retrieved chunks, and CV context to generate a comprehensive response that combines retrieved information with reasoning.",
                "benefits": "Produces human-like responses, synthesizes information from multiple sources, and provides contextual explanations."
            }
        }
        
        return explanations.get(step, {"title": "Unknown Step", "description": "", "technical_details": "", "benefits": ""})
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the entire RAG pipeline for educational purposes
        """
        return {
            "total_chunks": len(self.chunks),
            "chunk_size_avg": np.mean([chunk["word_count"] for chunk in self.chunks]) if self.chunks else 0,
            "embeddings_generated": len(self.embeddings),
            "embedding_dimension": len(self.embeddings[0]) if self.embeddings else 0,
            "pipeline_steps": [
                "Document Upload & Preprocessing",
                "Text Chunking with Overlap",
                "Embedding Generation",
                "Query Processing",
                "Semantic Retrieval",
                "Context Assembly",
                "LLM Response Generation"
            ]
        } 