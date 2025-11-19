"""
RAG (Retrieval-Augmented Generation) orchestration with hierarchical summarization.
Supports pluggable LLM backends (local fallback, OpenAI API, etc.).
"""

import logging
import os
from typing import List, Dict, Any, Optional, Callable
import re

logger = logging.getLogger(__name__)

# Try to import OpenAI (optional)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import Google Gemini (optional)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class RAGOrchestrator:
    """
    Orchestrates RAG pipeline: retrieval, prompt formatting, and LLM generation.
    """
    
    def __init__(
        self,
        vectorstore,
        embedder,
        top_k: int = 5,
        llm_provider: str = "local",  # "local", "openai", or custom
        llm_function: Optional[Callable] = None
    ):
        """
        Initialize RAG orchestrator.
        
        Args:
            vectorstore: VectorStore instance
            embedder: Embedder instance
            top_k: Number of chunks to retrieve
            llm_provider: LLM provider name ("local", "openai", "gemini", or custom)
            llm_function: Custom LLM function (text -> text)
        """
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.top_k = top_k
        self.llm_provider = llm_provider
        self.llm_function = llm_function
        
        # Initialize LLM based on provider
        if llm_provider == "openai":
            self._init_openai()
        elif llm_provider == "gemini":
            self._init_gemini()
        elif llm_provider == "local":
            self._init_local()
        elif llm_function:
            self._llm_call = llm_function
        else:
            logger.warning("No LLM provider specified, using local fallback")
            self._init_local()
    
    def _init_openai(self):
        """Initialize OpenAI API client."""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI not available, falling back to local")
            self._init_local()
            return
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, falling back to local")
            self._init_local()
            return
        
        # Set API key if using old-style API
        try:
            openai.api_key = api_key
        except:
            pass  # New API uses client initialization
        
        self._llm_call = self._openai_call
        logger.info("OpenAI LLM initialized")
    
    def _init_gemini(self):
        """Initialize Google Gemini API client."""
        if not GEMINI_AVAILABLE:
            logger.warning("Google Gemini not available, falling back to local")
            self._init_local()
            return
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set, falling back to local")
            self._init_local()
            return
        
        try:
            genai.configure(api_key=api_key)
            self._llm_call = self._gemini_call
            logger.info("Google Gemini LLM initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self._init_local()
    
    def _init_local(self):
        """Initialize local fallback summarizer."""
        self._llm_call = self._local_summarizer
        logger.info("Local fallback summarizer initialized")
    
    def _openai_call(self, prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 1000) -> str:
        """
        Call OpenAI API.
        
        Args:
            prompt: Input prompt
            model: Model name
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI not available, using local fallback")
            return self._local_summarizer(prompt)
        
        try:
            # Try new OpenAI API (v1.0+)
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except ImportError:
            # Fallback to old API style if available
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"OpenAI API call failed: {e}")
                return self._local_summarizer(prompt)
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            # Fallback to local
            return self._local_summarizer(prompt)
    
    def _gemini_call(
        self,
        prompt: str,
        images: Optional[List] = None,
        model: str = "gemini-pro",
        max_tokens: int = 1000
    ) -> str:
        """
        Call Google Gemini API with optional image support.
        
        Args:
            prompt: Input prompt
            images: Optional list of PIL Images or image paths
            model: Model name (gemini-pro, gemini-pro-vision, gemini-2.5-flash, etc.)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if not GEMINI_AVAILABLE:
            logger.warning("Gemini not available, using local fallback")
            return self._local_summarizer(prompt)
        
        try:
            # Auto-select vision model if images are provided
            if images and "vision" not in model.lower() and "2.5" not in model:
                # Use gemini-2.5-flash (latest, faster) which supports vision
                if "flash" in model.lower() or "fast" in model.lower():
                    model = "gemini-2.5-flash"
                else:
                    # Default to gemini-2.5-flash for vision
                    model = "gemini-2.5-flash"
                logger.info(f"Auto-selected vision model: {model}")
            
            # Build content list
            content = [prompt]
            
            # Add images if provided
            if images:
                try:
                    from PIL import Image as PILImage
                    for img in images:
                        if isinstance(img, str):
                            # Load image from path
                            img = PILImage.open(img)
                        elif isinstance(img, PILImage.Image):
                            # PIL Image object - already correct
                            pass
                        else:
                            logger.warning(f"Skipping invalid image type: {type(img)}")
                            continue
                        content.append(img)
                except ImportError:
                    logger.warning("PIL not available, skipping images")
                except Exception as e:
                    logger.warning(f"Failed to process images: {e}")
            
            # Try to use the model, with fallback if it fails
            try:
                model_instance = genai.GenerativeModel(model)
                response = model_instance.generate_content(
                    content,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.7,
                    )
                )
                # Handle blocked/filtered responses
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if candidate.finish_reason == 2:  # SAFETY (blocked)
                        logger.warning("Response was blocked by safety filters")
                        return "I apologize, but I cannot generate a response due to content safety filters. Please try rephrasing your query."
                    elif candidate.finish_reason == 3:  # RECITATION (blocked)
                        logger.warning("Response was blocked due to recitation concerns")
                        return "I apologize, but I cannot generate a response due to recitation concerns. Please try rephrasing your query."
                    elif candidate.content and candidate.content.parts:
                        return candidate.content.parts[0].text.strip()
                    else:
                        return "No response generated. Please try again."
                else:
                    return "No response generated. Please try again."
            except Exception as model_error:
                # If model doesn't exist, try alternatives
                error_str = str(model_error).lower()
                if "not found" in error_str or "404" in error_str or "not supported" in error_str:
                    logger.warning(f"Model {model} not available, trying fallback models...")
                    # Try gemini-pro-vision as fallback
                    fallback_models = ["gemini-2.5-flash", "gemini-pro-vision", "gemini-1.5-flash", "gemini-pro"]
                    for fallback_model in fallback_models:
                        if fallback_model == model:
                            continue
                        try:
                            logger.info(f"Trying fallback model: {fallback_model}")
                            model_instance = genai.GenerativeModel(fallback_model)
                            response = model_instance.generate_content(
                                content,
                                generation_config=genai.types.GenerationConfig(
                                    max_output_tokens=max_tokens,
                                    temperature=0.7,
                                )
                            )
                            logger.info(f"Successfully used fallback model: {fallback_model}")
                            return response.text.strip()
                        except Exception as fallback_error:
                            logger.debug(f"Fallback model {fallback_model} also failed: {fallback_error}")
                            continue
                # Re-raise if it's not a model not found error
                raise model_error
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            # Fallback to local
            return self._local_summarizer(prompt)
    
    def _local_summarizer(self, prompt: str) -> str:
        """
        Simple local summarizer (deterministic fallback).
        Extracts key sentences and formats them.
        
        Args:
            prompt: Input prompt with context
            
        Returns:
            Summarized text
        """
        # Extract the main query and context from prompt
        # This is a simple heuristic-based summarizer
        lines = prompt.split("\n")
        context_start = False
        context_text = []
        query = ""
        
        for line in lines:
            if "Context:" in line or "---" in line:
                context_start = True
                continue
            if context_start and line.strip():
                if "Query:" in line or "Question:" in line:
                    query = line.split(":", 1)[1].strip()
                    break
                context_text.append(line.strip())
        
        # Simple extraction: take first few sentences and key phrases
        full_context = " ".join(context_text)
        sentences = re.split(r'[.!?]+', full_context)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Return first 3-5 sentences as summary
        summary_sentences = sentences[:5]
        summary = ". ".join(summary_sentences)
        if summary and not summary.endswith("."):
            summary += "."
        
        if query:
            summary = f"Query: {query}\n\nSummary: {summary}"
        
        return summary or "Summary could not be generated."
    
    def retrieve(self, query: str, video_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            video_id: Optional video ID to filter results
            
        Returns:
            List of retrieved chunks with metadata
        """
        # Generate query embedding
        query_embedding = self.embedder.encode_single(query).tolist()
        
        # Build filter if video_id provided
        filter_dict = {"video_id": video_id} if video_id else None
        
        # Query vector store
        results = self.vectorstore.query(
            query_embedding=query_embedding,
            top_k=self.top_k,
            filter_dict=filter_dict
        )
        
        return results
    
    def format_prompt(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        prompt_type: str = "summary",
        visual_context: Optional[Dict[str, List]] = None
    ) -> str:
        """
        Format prompt with retrieved context and optional visual context.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved chunks
            prompt_type: Type of prompt ("summary", "notes", "qa")
            visual_context: Optional dict mapping chunk_id to list of frame dicts
            
        Returns:
            Formatted prompt string
        """
        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            metadata = chunk.get("metadata", {})
            start = metadata.get("start", 0.0)
            end = metadata.get("end", 0.0)
            text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id", "")
            
            chunk_info = f"[Chunk {i}, {self._format_timestamp(start)}-{self._format_timestamp(end)}]\n{text}"
            
            # Add visual context if available
            if visual_context and chunk_id in visual_context:
                frames = visual_context[chunk_id]
                if frames:
                    frame_times = [self._format_timestamp(f.get("timestamp", 0)) for f in frames]
                    chunk_info += f"\n[Visual content available at: {', '.join(frame_times)}]"
            
            context_parts.append(chunk_info)
        
        context = "\n\n".join(context_parts)
        
        # Build prompt based on type
        visual_instruction = ""
        if visual_context:
            visual_instruction = "\n\nNote: Visual content (frames/images) are provided alongside the text context. Please analyze both the transcript and the visual content to provide comprehensive answers."
        
        if prompt_type == "summary":
            prompt = f"""You are analyzing a video transcript. Use ALL the provided context chunks to create a comprehensive summary.

Context from video transcript ({len(retrieved_chunks)} chunks):
---
{context}
---{visual_instruction}

Query: {query}

Instructions:
- Synthesize information from ALL {len(retrieved_chunks)} provided chunks above
- Create a coherent, comprehensive summary that addresses the query
- Combine and synthesize information across chunks - do not just repeat individual chunks
- Reference specific timestamps when relevant
- If visual content is mentioned, incorporate information from both transcript and images

Summary:"""
        
        elif prompt_type == "notes":
            prompt = f"""You are creating structured notes from a video transcript. Use ALL the provided context chunks.

Context from video transcript ({len(retrieved_chunks)} chunks):
---
{context}
---{visual_instruction}

Query: {query}

Instructions:
- Synthesize information from ALL {len(retrieved_chunks)} provided chunks
- Create well-organized, structured notes with key points and main ideas
- Combine information across chunks - do not just list individual chunks
- Include timestamps where relevant
- Organize notes logically with clear sections

Please produce structured notes with the following sections:
1. Overview
2. Key Concepts
3. Examples
4. Action Items
5. References (with timestamps)

Format the response clearly with bullet points. If visual content is provided, describe relevant visual elements (slides, diagrams, charts, etc.) in the appropriate sections."""
        
        else:  # qa or default
            prompt = f"""You are answering questions about a video transcript. Use ALL the provided context chunks to give a comprehensive answer.

Context from video transcript ({len(retrieved_chunks)} chunks):
---
{context}
---{visual_instruction}

Query: {query}

Instructions:
- Use information from ALL {len(retrieved_chunks)} provided chunks to answer the question
- Synthesize information across chunks to provide a complete answer
- Be specific and reference timestamps where relevant
- If the answer requires information from multiple chunks, combine them coherently

Answer:"""
        
        return prompt
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def generate(
        self,
        query: str,
        video_id: Optional[str] = None,
        prompt_type: str = "summary",
        visual_context: Optional[Dict[str, List]] = None
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline: retrieve, format, generate with optional visual context.
        
        Args:
            query: User query
            video_id: Optional video ID filter
            prompt_type: Prompt type ("summary", "notes", "qa")
            visual_context: Optional dict mapping chunk_id to list of frame dicts with 'image' key
            
        Returns:
            Dict with keys: summary, evidence (retrieved chunks), query
        """
        logger.info(f"RAG generation: query='{query[:50]}...', video_id={video_id}")
        
        # Retrieve
        retrieved_chunks = self.retrieve(query, video_id)
        
        if not retrieved_chunks:
            return {
                "summary": "No relevant content found.",
                "evidence": [],
                "query": query
            }
        
        # Format prompt
        prompt = self.format_prompt(query, retrieved_chunks, prompt_type, visual_context)
        
        # Collect images from visual context for relevant chunks
        images = []
        if visual_context:
            for chunk in retrieved_chunks:
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id in visual_context:
                    for frame in visual_context[chunk_id]:
                        if "image" in frame:
                            images.append(frame["image"])
        
        # Generate with images if available
        if images and self.llm_provider == "gemini":
            summary = self._gemini_call(prompt, images=images)
        else:
            summary = self._llm_call(prompt)
        
        return {
            "summary": summary,
            "evidence": retrieved_chunks,
            "query": query
        }
    
    def hierarchical_summarize(
        self,
        chunks: List[Dict[str, Any]],
        video_id: str
    ) -> Dict[str, Any]:
        """
        Hierarchical summarization: micro-summaries -> meta-summary.
        
        Args:
            chunks: List of chunks to summarize
            video_id: Video identifier
            
        Returns:
            Dict with micro_summaries and meta_summary
        """
        logger.info(f"Hierarchical summarization for {len(chunks)} chunks")
        
        # Generate micro-summaries for each chunk
        micro_summaries = []
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            if chunk_text:
                micro_prompt = f"Summarize this content in 1-2 sentences:\n\n{chunk_text}"
                micro_summary = self._llm_call(micro_prompt)
                micro_summaries.append({
                    "chunk_id": chunk.get("chunk_id", ""),
                    "start": chunk.get("start", 0.0),
                    "end": chunk.get("end", 0.0),
                    "summary": micro_summary
                })
        
        # Generate meta-summary from micro-summaries
        micro_text = "\n\n".join([
            f"[{self._format_timestamp(ms['start'])}-{self._format_timestamp(ms['end'])}] {ms['summary']}"
            for ms in micro_summaries
        ])
        
        meta_prompt = f"""Create a comprehensive summary from these micro-summaries:

{micro_text}

Provide a structured overview with key themes and insights."""
        
        meta_summary = self._llm_call(meta_prompt)
        
        return {
            "micro_summaries": micro_summaries,
            "meta_summary": meta_summary,
            "video_id": video_id
        }

