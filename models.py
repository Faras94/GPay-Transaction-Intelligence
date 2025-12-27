from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class Transaction:
    """
    Data Transfer Object representing a single financial transaction.
    This is the core contract between Ingestion, Storage, and Analytics layers.
    """
    id: str  # Unique hash of the transaction
    date: datetime
    description: str
    amount: float
    type: str  # 'Spent' or 'Received'
    category: str
    source_file: str
    time: Optional[str] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame compatibility."""
        return {
            'id': self.id,
            'date': self.date,
            'description': self.description,
            'amount': self.amount,
            'type': self.type,
            'category': self.category,
            'time': self.time,
            'source_file': self.source_file,
            **self.raw_metadata
        }

@dataclass
class IngestionResult:
    """Result of an ingestion process."""
    transactions: list[Transaction]
    file_path: str
    success: bool
    error_message: Optional[str] = None
