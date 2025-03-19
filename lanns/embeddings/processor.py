"""
Data processors for LANNS embeddings.
"""
import os
import json
import time
import gc
from typing import List, Dict, Any, Optional, Union, Tuple, Iterator, Generator
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Base class for data processors"""
    
    def __init__(self, 
                file_path: str,
                batch_size: int = 1024,
                id_field: str = "ROW_ID"):
        self.file_path = file_path
        self.batch_size = batch_size
        self.id_field = id_field
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
    
    def process(self):
        """Process data and yield batches"""
        raise NotImplementedError("Subclasses must implement process()")
    
    def _format_record_for_embedding(self, record, fields_order=None):
        """
        Format a record dictionary into a string suitable for embedding.
        Handles missing fields gracefully.
        """
        if fields_order is None:
            # Default ordering of fields
            fields_order = [
                "FIRST_NAME", "MIDDLE_NAME", "LAST_NAME", "SUFFIX",
                "DOB", "EMAIL", "PHONE", 
                "ADDRESS", "CITY", "STATE", "ZIP",
                "VIN", "DATAFILE_ID"
            ]
        
        # Filter out internal fields we probably don't want to embed
        exclude_fields = ["ROW_ID", "RAND"]
        
        # Build parts list from available fields
        parts = []
        
        # First add fields in the specified order
        for field in fields_order:
            if field in record and field not in exclude_fields:
                value = record[field]
                if value:  # Skip empty values
                    parts.append(f"{field.lower()}: {value}")
        
        # Then add any remaining fields not in the specified order
        for field, value in record.items():
            if (field not in fields_order and 
                field not in exclude_fields and 
                value):  # Skip empty values
                parts.append(f"{field.lower()}: {value}")
        
        # Join into a single string
        return " | ".join(parts)

class NDJSONProcessor(DataProcessor):
    """Processor for NDJSON files with streaming support"""
    
    def __init__(self, 
                file_path: str,
                batch_size: int = 1024,
                id_field: str = "ROW_ID",
                text_field: Optional[str] = None,
                fields_order: Optional[List[str]] = None,
                prompt_prefix: Optional[str] = None):
        """
        Initialize NDJSON processor.
        
        Args:
            file_path: Path to NDJSON file
            batch_size: Number of records per batch
            id_field: Field to use as ID
            text_field: Field containing text to embed (if None, format entire record)
            fields_order: List of fields to prioritize in formatting
            prompt_prefix: Optional prefix to add to each text
        """
        super().__init__(file_path, batch_size, id_field)
        self.text_field = text_field
        self.fields_order = fields_order
        self.prompt_prefix = prompt_prefix
        
        # Count lines for progress tracking
        self.total_lines = self._count_lines()
        logger.info(f"Found {self.total_lines:,} lines in {file_path}")
    
    def _count_lines(self):
        """Count lines in file efficiently"""
        # Try using wc -l on Unix systems
        if os.name != 'nt':  # Not Windows
            try:
                import subprocess
                result = subprocess.run(['wc', '-l', self.file_path], 
                                     capture_output=True, text=True)
                if result.returncode == 0:
                    return int(result.stdout.strip().split()[0])
            except:
                pass
        
        # Fallback to manual counting
        count = 0
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
        return count
    
    def process(self):
        """
        Process NDJSON file and yield batches.
        
        Yields:
            tuple: (batch_texts, batch_ids, batch_idx, is_last_batch)
        """
        batch_texts = []
        batch_ids = []
        batch_data = []
        batch_idx = 0
        
        with tqdm(total=self.total_lines, desc="Reading NDJSON") as pbar:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    pbar.update(1)
                    
                    if not line.strip():
                        continue
                    
                    try:
                        record = json.loads(line)
                        
                        if isinstance(record, dict):
                            # Get ID (or use line number if not available)
                            item_id = record.get(self.id_field, line_num)
                            
                            # Get or format text
                            if self.text_field is not None and self.text_field in record:
                                # Use the specified text field
                                text = record[self.text_field]
                            else:
                                # Format the entire record
                                text = self._format_record_for_embedding(record, self.fields_order)
                            
                            # Add prompt prefix if specified
                            if self.prompt_prefix:
                                text = f"{self.prompt_prefix}{text}"
                            
                            batch_data.append(record)
                            batch_texts.append(text)
                            batch_ids.append(item_id)
                            
                            # When we reach batch size, yield the batch
                            if len(batch_texts) >= self.batch_size:
                                is_last = line_num == self.total_lines - 1
                                yield batch_data, batch_texts, batch_ids, batch_idx, is_last
                                batch_data = []
                                batch_texts = []
                                batch_ids = []
                                batch_idx += self.batch_size
                                
                                # Manual garbage collection to free memory
                                gc.collect()
                        
                    except json.JSONDecodeError:
                        pbar.write(f"Warning: Could not parse line {line_num} as JSON")
        
        # Yield the final batch if there's anything left
        if batch_texts:
            yield batch_data, batch_texts, batch_ids, batch_idx, True

# Factory function to get the right processor
def get_processor(file_path, format=None, **kwargs):
    """Get appropriate processor based on file format"""
    if format is None:
        # Auto-detect format from file extension
        if file_path.endswith('.ndjson') or file_path.endswith('.jsonl'):
            format = 'ndjson'
        elif file_path.endswith('.json'):
            format = 'json'
        elif file_path.endswith('.csv'):
            format = 'csv'
        elif file_path.endswith('.txt'):
            format = 'text'
        else:
            raise ValueError(f"Could not detect format for {file_path}")
    
    if format == 'ndjson':
        return NDJSONProcessor(file_path, **kwargs)
    else:
        # Implement other processors as needed
        raise ValueError(f"Unsupported format: {format}")