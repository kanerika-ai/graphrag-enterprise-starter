#!/usr/bin/env python3
"""
GraphRAG vs Flat RAG — Side-by-Side Comparison

Run this script to:
1. Ingest PDF documents from data/pdfs/
2. Index using both flat vector RAG and graph-enhanced RAG
3. Run 10 evaluation questions through both pipelines
4. Score and compare the results

Prerequisites:
    pip install -r requirements.txt
    # Models run locally via transformers — no API key needed
    python data/generate_pdfs.py # generate the PDF dataset

Usage:
    python run_comparison.py
"""

import os
import sys
import time

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm import check_hf_available, get_model
from src.ingest import load_pdfs
from src.flat_rag import FlatRAG
from src.graph_rag import GraphRAG
from src.compare import run_comparison, print_summary, save_results


def main():
    console = Console()

    model_name = get_model()
    console.print(Panel.fit(
        "[bold]GraphRAG vs Flat RAG — Enterprise Knowledge Base Comparison[/bold]\n"
        f"Company: Arcturus Systems Inc. | LLM: {model_name} (HuggingFace) | Embeddings: sentence-transformers\n"
        "Data source: PDF documents | 10 evaluation questions across 4 categories",
        border_style="cyan",
    ))

    # ---- Pre-flight checks ----
    ok, msg = check_hf_available()
    if not ok:
        console.print(f"[red]Error: {msg}[/red]")
        sys.exit(1)
    console.print(f"[green]  {msg}[/green]")

    # ---- Step 0: Load PDFs ----
    console.print("\n[bold]Step 0: Ingesting PDF documents[/bold]")
    try:
        documents = load_pdfs()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    console.print(f"  Loaded {len(documents)} documents from data/pdfs/")

    # ---- Step 1: Index with Flat RAG ----
    console.print("\n[bold blue]Step 1: Indexing with Flat Vector RAG (ChromaDB + sentence-transformers)[/bold blue]")
    t0 = time.time()
    flat_rag = FlatRAG()
    # num_chunks = flat_rag.index(documents)
    # flat_index_time = time.time() - t0
    # console.print(f"  Indexed {num_chunks} chunks in {flat_index_time:.1f}s")

    # ---- Step 2: Index with Graph RAG ----
    console.print(f"\n[bold green]Step 2: Indexing with Graph RAG (entity extraction via HF:{model_name} + NetworkX)[/bold green]")
    t0 = time.time()
    graph_rag = GraphRAG()
    # stats = graph_rag.index(documents, verbose=True)
    # graph_index_time = time.time() - t0
    # console.print(f"  Built graph with {stats['num_entities']} entities, "
    #               f"{stats['num_relationships']} relationships, "
    #               f"{stats['num_communities']} communities in {graph_index_time:.1f}s")

    # ---- Step 3: Run comparison ----
    console.print(f"\n[bold yellow]Step 3: Running evaluation questions through both pipelines[/bold yellow]")
    results = run_comparison(flat_rag, graph_rag, verbose=True)

    # ---- Step 4: Print summary ----
    console.print("\n" + "=" * 80)
    print_summary(results)

    # ---- Step 5: Save results ----
    save_results(results)

    # ---- Timing summary ----
    console.print(Panel.fit(
        f"[bold]Indexing Time[/bold]\n"
        f"  Flat RAG:  {flat_index_time:.1f}s (embed chunks with sentence-transformers)\n"
        f"  Graph RAG: {graph_index_time:.1f}s (HF API extraction + embed + community detection)\n\n"
        f"[dim]Graph RAG indexing is slower because it calls the HF Inference API for every document.\n"
        f"This is a one-time cost that pays off at query time for complex questions.[/dim]",
        border_style="yellow",
        title="Performance Notes",
    ))


if __name__ == "__main__":
    main()
