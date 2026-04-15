"""
Module variant consolidation.

Consolidates modules with the same base code (e.g., ACC1701A, ACC1701B → ACC1701)
into single representative modules with merged skills and data.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict

import pandas as pd


def extract_base_code(module_code: str) -> str:
    """Extract numeric base from module code (remove trailing letters)."""
    return re.sub(r'[A-Z]$', '', module_code)


def consolidate_module_variants(modules: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate module variants into single representative modules.
    
    Modules with the same base code (e.g., ACC1701A, ACC1701B, ..., ACC1701D)
    are merged into a single module with the base code (ACC1701).
    
    Consolidation strategy:
    - module_code: normalize to base (remove trailing letters)
    - Text fields (title, description): keep first value
    - List fields (skills, cues): merge and deduplicate across variants
    - Other fields: keep first value
    
    Args:
        modules: DataFrame with module data
        
    Returns:
        Consolidated DataFrame with no module variants
    """
    
    # Group every suffix variant under the same base module code first.
    base_code_groups: Dict[str, list] = defaultdict(list)
    for idx, row in modules.iterrows():
        base = extract_base_code(row['module_code'])
        base_code_groups[base].append((idx, row))
    
    consolidated_rows = []
    
    for base_code in sorted(base_code_groups.keys()):
        variant_data = base_code_groups[base_code]
        
        if len(variant_data) == 1:
            # No variants, just normalize the code
            row = variant_data[0][1].copy()
            row['module_code'] = base_code
            consolidated_rows.append(row)
        else:
            # Multiple variants - consolidate them
            # Start with the first variant as the representative row, then merge the
            # list-valued fields across all suffix variants.
            consolidated = variant_data[0][1].copy()
            consolidated['module_code'] = base_code
            
            # Merge list fields from all variants
            list_fields = ['technical_skills', 'soft_skills']
            for field in list_fields:
                if field in modules.columns:
                    all_items = []
                    for _, row in variant_data:
                        items = row.get(field)
                        if items is not None:
                            try:
                                # Handle both list and numpy array types
                                if hasattr(items, '__iter__') and not isinstance(items, str):
                                    all_items.extend(list(items))
                            except (TypeError, ValueError):
                                pass
                    
                    # Deduplicate while preserving order  
                    seen = set()
                    unique_items = []
                    for item in all_items:
                        if item not in seen:
                            seen.add(item)
                            unique_items.append(item)
                    
                    consolidated[field] = unique_items if unique_items else []
            
            consolidated_rows.append(consolidated)
    
    return pd.DataFrame(consolidated_rows).reset_index(drop=True)
