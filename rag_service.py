import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import logging

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            logger.info("OpenAI client initialized successfully")
        else:
            self.openai_client = None
            logger.warning("No OpenAI API key found - using TF-IDF fallback")
        
        self.chunks = []
        self.embeddings = []
        self.chunk_metadata = []
        self.vectorizer = None  # Store the vectorizer for TF-IDF queries
        self.use_openai = False  # Track which embedding method was used
        
    def chunk_document(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Split document into overlapping chunks with metadata
        Optimized for CV documents which are typically shorter and more structured
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Find the actual position in the original text
            start_pos = text.find(" ".join(chunk_words[:3])) if len(chunk_words) >= 3 else 0
            if start_pos == -1:
                start_pos = 0
            
            chunk_info = {
                "id": len(chunks),
                "text": chunk_text,
                "start_word": i,
                "end_word": min(i + chunk_size, len(words)),
                "word_count": len(chunk_words),
                "start_char": start_pos,
                "end_char": start_pos + len(chunk_text)
            }
            chunks.append(chunk_info)
            
        self.chunks = chunks
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Generate embeddings for each chunk using OpenAI API or TF-IDF fallback
        """
        if not chunks:
            logger.warning("No chunks provided for embedding generation")
            return []
        
        if self.openai_client:
            try:
                return self._generate_openai_embeddings(chunks)
            except Exception as e:
                logger.error(f"OpenAI embedding failed, falling back to TF-IDF: {e}")
                return self._generate_tfidf_embeddings(chunks)
        else:
            return self._generate_tfidf_embeddings(chunks)
    
    def _generate_openai_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Generate embeddings using OpenAI API
        """
        embeddings = []
        for i, chunk in enumerate(chunks):
            try:
                response = self.openai_client.embeddings.create(
                    input=chunk["text"],
                    model="text-embedding-ada-002"
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
                logger.debug(f"Generated OpenAI embedding for chunk {i}")
            except Exception as e:
                logger.error(f"Error generating OpenAI embedding for chunk {i}: {e}")
                # If any chunk fails, fall back to TF-IDF for all
                return self._generate_tfidf_embeddings(chunks)
        
        self.embeddings = embeddings
        self.use_openai = True
        logger.info(f"Generated {len(embeddings)} OpenAI embeddings")
        return embeddings
    
    def _generate_tfidf_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Generate TF-IDF embeddings as fallback
        """
        texts = [chunk["text"] for chunk in chunks]
        
        # Initialize and fit the vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000, 
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better semantic capture
            min_df=1,  # Include terms that appear in at least 1 document
            max_df=0.8  # Exclude terms that appear in more than 80% of documents
        )
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            embeddings = tfidf_matrix.toarray().tolist()
            
            self.embeddings = embeddings
            self.use_openai = False
            logger.info(f"Generated {len(embeddings)} TF-IDF embeddings with {len(self.vectorizer.get_feature_names_out())} features")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating TF-IDF embeddings: {e}")
            # Return zero embeddings as last resort
            embedding_dim = 100  # Default dimension
            embeddings = [[0.0] * embedding_dim for _ in chunks]
            self.embeddings = embeddings
            self.use_openai = False
            return embeddings
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query using the same method as the chunks
        """
        if self.use_openai and self.openai_client:
            try:
                response = self.openai_client.embeddings.create(
                    input=query,
                    model="text-embedding-ada-002"
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"Error generating OpenAI query embedding: {e}")
                # Fall back to TF-IDF if OpenAI fails
                self.use_openai = False
        
        # Use TF-IDF for query
        if self.vectorizer:
            try:
                query_tfidf = self.vectorizer.transform([query])
                return query_tfidf.toarray()[0].tolist()
            except Exception as e:
                logger.error(f"Error generating TF-IDF query embedding: {e}")
                # Return zero vector as last resort
                return [0.0] * len(self.embeddings[0]) if self.embeddings else [0.0] * 100
        else:
            logger.error("No vectorizer available for query embedding")
            return [0.0] * len(self.embeddings[0]) if self.embeddings else [0.0] * 100
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant chunks based on query similarity
        """
        if not self.embeddings:
            logger.warning("No embeddings available for retrieval")
            return []
        
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        try:
            # Generate query embedding using the same method as chunks
            query_embedding = self._get_query_embedding(query)
            
            if not query_embedding or all(x == 0 for x in query_embedding):
                logger.warning("Failed to generate valid query embedding")
                return []
            
            # Calculate similarities
            similarities = []
            for i, chunk_embedding in enumerate(self.embeddings):
                try:
                    # Ensure both embeddings have the same dimension
                    if len(query_embedding) != len(chunk_embedding):
                        logger.warning(f"Dimension mismatch: query({len(query_embedding)}) vs chunk({len(chunk_embedding)})")
                        continue
                    
                    similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                    similarities.append((i, float(similarity)))
                except Exception as e:
                    logger.error(f"Error calculating similarity for chunk {i}: {e}")
                    continue
            
            if not similarities:
                logger.warning("No valid similarities calculated")
                return []
            
            # Sort by similarity and get top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_chunks = []
            
            for rank, (chunk_idx, similarity) in enumerate(similarities[:top_k]):
                if chunk_idx < len(self.chunks):
                    chunk_info = self.chunks[chunk_idx].copy()
                    chunk_info["similarity_score"] = similarity
                    chunk_info["rank"] = rank + 1
                    top_chunks.append(chunk_info)
            
            logger.info(f"Retrieved {len(top_chunks)} relevant chunks for query: '{query[:50]}...'")
            return top_chunks
            
        except Exception as e:
            logger.error(f"Error in retrieve_relevant_chunks: {e}")
            return []
    
    def generate_rag_response(self, query: str, retrieved_chunks: List[Dict[str, Any]], 
                            cv_text: str) -> Dict[str, Any]:
        """
        Generate RAG response using LLM with retrieved context
        """
        if not retrieved_chunks:
            return {
                "answer": "I couldn't find relevant information in the CV to answer your question. Please try rephrasing your query or ask about specific aspects like skills, experience, or education.",
                "query": query,
                "retrieved_chunks": [],
                "context_used": ""
            }
        
        # Prepare context from retrieved chunks
        context_parts = []
        for chunk in retrieved_chunks:
            context_parts.append(f"Relevant Section (Similarity: {chunk['similarity_score']:.3f}):\n{chunk['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for LLM
        prompt = f"""Based on the following CV information and retrieved relevant sections, answer the question clearly and accurately.

Question: {query}

Retrieved Relevant Sections:
{context}

Instructions:
- Provide a direct, comprehensive answer based on the retrieved information
- If the information is incomplete, indicate what additional details might be helpful
- Be specific and reference relevant details from the CV
- If no relevant information is found, say so clearly

Answer:"""

        try:
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful CV analysis assistant. Provide clear, accurate, succinct answers based on the given CV context. Be specific and reference relevant details. No more than 3 sentences."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.3  # Lower temperature for more consistent responses
                )
                answer = response.choices[0].message.content.strip()
                logger.info("Generated OpenAI response successfully")
            else:
                # Fallback response with extracted information
                answer = f"Based on the retrieved CV sections:\n\n{context}\n\nRegarding your question '{query}': The most relevant information has been extracted above. For a more detailed analysis, please provide an OpenAI API key."
                logger.info("Generated fallback response")
                
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            answer = f"I found relevant information in the CV but encountered an error generating a detailed response. Here's what I found:\n\n{context}"
        
        return {
            "answer": answer,
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "context_used": context
        }
    
    def _build_embedding_previews(self, max_values: int = 8) -> List[Dict[str, Any]]:
        """
        Build embedding previews for educational display
        """
        previews = []
        if not self.embeddings:
            return previews
            
        for i, embedding in enumerate(self.embeddings[:3]):  # Show only first 3 chunks
            try:
                # Ensure numeric and round for display
                head_values = []
                for val in embedding[:max_values]:
                    try:
                        head_values.append(round(float(val), 4))
                    except (ValueError, TypeError):
                        head_values.append(0.0)
                        
                previews.append({
                    "chunk_id": i,
                    "values": head_values
                })
            except Exception as e:
                logger.error(f"Error building preview for embedding {i}: {e}")
                continue
                
        return previews
    
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
        try:
            embedding_dimension = len(self.embeddings[0]) if self.embeddings else 0
            chunk_size_avg = np.mean([chunk["word_count"] for chunk in self.chunks]) if self.chunks else 0
            
            return {
                "total_chunks": len(self.chunks),
                "chunk_size_avg": float(chunk_size_avg),
                "embeddings_generated": len(self.embeddings),
                "embedding_dimension": embedding_dimension,
                "embedding_method": "OpenAI" if self.use_openai else "TF-IDF",
                "embedding_previews": self._build_embedding_previews(),
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
        except Exception as e:
            logger.error(f"Error generating pipeline summary: {e}")
            return {
                "total_chunks": len(self.chunks),
                "chunk_size_avg": 0,
                "embeddings_generated": len(self.embeddings),
                "embedding_dimension": 0,
                "embedding_method": "Unknown",
                "embedding_previews": [],
                "pipeline_steps": []
            }