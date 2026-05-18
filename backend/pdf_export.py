"""
PDF export functionality using FPDF for generating formatted notes.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

logger = logging.getLogger(__name__)

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    logger.warning("fpdf2 not available. Install with: pip install fpdf2")


class PDFExporter:
    """
    Generate PDF notes from summaries and evidence.
    """
    
    def __init__(self):
        """Initialize PDF exporter."""
        if not FPDF_AVAILABLE:
            raise ImportError(
                "fpdf2 is not installed. "
                "Install with: pip install fpdf2"
            )
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def _plain_text(self, text: str) -> str:
        """Convert common Markdown output into clean PDF text."""
        if not text:
            return ""

        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"```(?:\w+)?\n([\s\S]*?)```", r"\1", cleaned)
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
        cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"(?<!\w)(\*\*|__)(.+?)\1(?!\w)", r"\2", cleaned)
        cleaned = re.sub(r"(?<!\w)(\*|_)(.+?)\1(?!\w)", r"\2", cleaned)
        cleaned = re.sub(r"^\s*[-*+]\s+", "- ", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
    
    def export_notes(
        self,
        output_path: str,
        title: str,
        summary: str,
        evidence: Optional[List[Dict[str, Any]]] = None,
        query: Optional[str] = None,
        video_id: Optional[str] = None,
        include_metadata: bool = False,
        include_evidence: bool = False,
    ) -> bool:
        """
        Export notes to PDF.
        
        Args:
            output_path: Path to output PDF file
            title: Document title
            summary: Generated summary text
            evidence: List of evidence chunks with metadata
            query: Original query (optional)
            video_id: Video identifier (optional)
            include_metadata: Whether to include query/video_id/generated timestamp
            include_evidence: Whether to include an evidence/references section
            
        Returns:
            True if successful
        """
        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # Add first page
            pdf.add_page()
            
            # Title
            pdf.set_font("Arial", "B", 20)
            pdf.cell(0, 10, title, ln=1, align="C")
            pdf.ln(5)

            # Metadata (optional). Default is OFF to keep PDFs presentation-ready.
            if include_metadata:
                pdf.set_font("Arial", "", 10)
                if video_id:
                    pdf.cell(0, 5, f"Video ID: {video_id}", ln=1)
                pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
                if query:
                    pdf.cell(0, 5, f"Query: {query}", ln=1)
                pdf.ln(5)
            
            # Notes section (clean, no references)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 8, "Notes", ln=1)
            pdf.set_font("Arial", "", 11)
            
            # Split summary into lines that fit page width
            summary_lines = self._wrap_text(self._plain_text(summary), pdf.w - 40)
            for line in summary_lines:
                pdf.cell(0, 6, line, ln=1)
            
            pdf.ln(5)
            
            # Evidence section
            if include_evidence and evidence:
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 8, "Evidence & References", ln=1)
                pdf.set_font("Arial", "", 10)
                
                for i, chunk in enumerate(evidence, 1):
                    metadata = chunk.get("metadata", {})
                    start = metadata.get("start", 0.0)
                    end = metadata.get("end", 0.0)
                    text = chunk.get("text", "")
                    
                    # Chunk header
                    pdf.set_font("Arial", "B", 11)
                    pdf.cell(0, 6, f"Reference {i} [{self._format_timestamp(start)} - {self._format_timestamp(end)}]", ln=1)
                    pdf.set_font("Arial", "", 10)
                    
                    # Chunk text
                    text_lines = self._wrap_text(self._plain_text(text), pdf.w - 40)
                    for line in text_lines:
                        pdf.cell(0, 5, line, ln=1)
                    
                    pdf.ln(3)
            
            # Save PDF
            pdf.output(output_path)
            logger.info(f"PDF exported: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"PDF export failed: {e}")
            raise
    
    def _wrap_text(self, text: str, max_width: float) -> List[str]:
        """
        Wrap text to fit PDF page width.
        
        Args:
            text: Text to wrap
            max_width: Maximum width in PDF units
            
        Returns:
            List of wrapped lines
        """
        lines = []

        for paragraph in text.split("\n"):
            if not paragraph.strip():
                lines.append("")
                continue

            words = paragraph.split()
            current_line = ""

            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                # Approximate: 1 character is about 2 PDF units for Arial 10pt.
                if len(test_line) * 2 <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)

        return lines if lines else [text]
    
    def export_hierarchical_notes(
        self,
        output_path: str,
        title: str,
        hierarchical_result: Dict[str, Any]
    ) -> bool:
        """
        Export hierarchical summarization results to PDF.
        
        Args:
            output_path: Path to output PDF
            title: Document title
            hierarchical_result: Result from hierarchical_summarize()
            
        Returns:
            True if successful
        """
        meta_summary = hierarchical_result.get("meta_summary", "")
        micro_summaries = hierarchical_result.get("micro_summaries", [])
        video_id = hierarchical_result.get("video_id", "")
        
        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            
            # Title
            pdf.set_font("Arial", "B", 20)
            pdf.cell(0, 10, title, ln=1, align="C")
            pdf.ln(5)
            
            # Metadata
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 5, f"Video ID: {video_id}", ln=1)
            pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
            pdf.ln(5)
            
            # Meta Summary
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 8, "Overview", ln=1)
            pdf.set_font("Arial", "", 11)
            summary_lines = self._wrap_text(self._plain_text(meta_summary), pdf.w - 40)
            for line in summary_lines:
                pdf.cell(0, 6, line, ln=1)
            
            pdf.ln(5)
            
            # Micro Summaries
            if micro_summaries:
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 8, "Section Summaries", ln=1)
                pdf.set_font("Arial", "", 10)
                
                for ms in micro_summaries:
                    pdf.set_font("Arial", "B", 11)
                    pdf.cell(0, 6, f"[{self._format_timestamp(ms['start'])} - {self._format_timestamp(ms['end'])}]", ln=1)
                    pdf.set_font("Arial", "", 10)
                    summary_lines = self._wrap_text(self._plain_text(ms['summary']), pdf.w - 40)
                    for line in summary_lines:
                        pdf.cell(0, 5, line, ln=1)
                    pdf.ln(3)
            
            pdf.output(output_path)
            logger.info(f"Hierarchical PDF exported: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Hierarchical PDF export failed: {e}")
            raise
