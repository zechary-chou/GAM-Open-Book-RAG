# -*- coding: utf-8 -*-
"""
Retriever Module

This module contains all retrieval implementations for the GAM framework.
Retrievers provide different search strategies for finding relevant information.

Available Retrievers:
- DenseRetriever: Semantic search using dense vector embeddings
- BM25Retriever: Keyword-based search using BM25 algorithm
- IndexRetriever: Direct page access by index
"""

from __future__ import annotations

from .base import AbsRetriever
from .dense_retriever import DenseRetriever
from .bm25 import BM25Retriever
from .index_retriever import IndexRetriever

__all__ = [
    "AbsRetriever",
    "DenseRetriever",
    "BM25Retriever", 
    "IndexRetriever",
]
