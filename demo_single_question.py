#!/usr/bin/env python3
"""
Quick demo: ask a single question to both pipelines and compare.

Usage:
    python demo_single_question.py "Who does Dr. Anika Patel collaborate with?"
    python demo_single_question.py  # uses a default multi-hop question
"""

import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm import check_hf_available
from src.ingest import load_pdfs
from src.flat_rag import FlatRAG
from src.graph_rag import GraphRAG


def main():
    console = Console()

    question = (
        sys.argv[1] if len(sys.argv) > 1
        else "If Project Atlas is delayed, what downstream projects and deals would be affected and why?"
    )

    ok, msg = check_hf_available()
    if not ok:
        console.print(f"[red]{msg}[/red]")
        sys.exit(1)

    console.print(Panel.fit(f"[bold]Question:[/bold] {question}", border_style="cyan"))

    # Load PDFs
    console.print("\n[dim]Loading PDF documents...[/dim]")
    try:
        documents = load_pdfs()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    console.print(f"[dim]Loaded {len(documents)} documents.[/dim]")

    # Index both
    console.print("[dim]Indexing (one-time)...[/dim]")
    flat_rag = FlatRAG()
    flat_rag.index(documents)

    graph_rag = GraphRAG()
    graph_rag.index(documents, verbose=False)
    console.print("[dim]Done.[/dim]\n")

    # Query both
    flat_result = flat_rag.query(question)
    graph_result = graph_rag.query(question)

    console.print(Panel(
        flat_result["answer"],
        title="[blue]Flat Vector RAG Answer[/blue]",
        subtitle=f"Sources: {', '.join(flat_result['sources'][:5])}",
        border_style="blue",
    ))

    console.print(Panel(
        graph_result["answer"],
        title="[green]Graph-Enhanced RAG Answer[/green]",
        subtitle=f"Seed entities: {', '.join(graph_result.get('seed_entities', [])[:5])}",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
