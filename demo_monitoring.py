#!/usr/bin/env python3
"""
Demo script for the GPT-OSS Hallucination Monitor.
This demonstrates the functionality without requiring heavy dependencies.
"""

import json
import tempfile
import os

def create_demo_files():
    """Create demo files for testing."""
    
    # Create prompt file
    with open("demo_prompt.txt", "w") as f:
        f.write("What is the capital of France?")
    
    # Create completion file
    with open("demo_completion.txt", "w") as f:
        f.write("Paris is the capital of France with a population of 2.2 million people.")
    
    # Create context file
    with open("demo_context.txt", "w") as f:
        f.write("Paris is the capital and most populous city of France.")
    
    print("✓ Demo files created")

def show_cli_usage():
    """Show CLI usage examples."""
    print("\n" + "="*60)
    print("GPT-OSS HALLUCINATION MONITOR - CLI USAGE")
    print("="*60)
    
    print("\n1. Basic Usage:")
    print("   gpt-oss-monitor --prompt demo_prompt.txt --completion demo_completion.txt")
    
    print("\n2. With Context Documents:")
    print("   gpt-oss-monitor --prompt demo_prompt.txt --completion demo_completion.txt --contexts demo_context.txt")
    
    print("\n3. Generate HTML Report:")
    print("   gpt-oss-monitor --prompt demo_prompt.txt --completion demo_completion.txt --html")
    
    print("\n4. Custom Configuration:")
    print("   gpt-oss-monitor --prompt demo_prompt.txt --completion demo_completion.txt --k 10 --temperature 0.8 --html")
    
    print("\n5. Save JSON Results:")
    print("   gpt-oss-monitor --prompt demo_prompt.txt --completion demo_completion.txt --output results.json")

def show_python_api():
    """Show Python API examples."""
    print("\n" + "="*60)
    print("GPT-OSS HALLUCINATION MONITOR - PYTHON API")
    print("="*60)
    
    print("\n1. Basic Usage:")
    print("""
from gpt_oss.monitoring import HallucinationMonitor

# Initialize monitor
monitor = HallucinationMonitor()

# Evaluate a completion
results = monitor.evaluate(
    prompt="What is the capital of France?",
    completion="Paris is the capital of France with 2.2 million people."
)

print(f"Risk Level: {results['risk_level']}")
print(f"Risk Score: {results['risk_score']:.3f}")
""")
    
    print("\n2. With Context Documents:")
    print("""
context_docs = [
    "Paris is the capital and most populous city of France.",
    "The population of Paris was 2,165,423 in 2019."
]

results = monitor.evaluate(
    prompt="What is the capital of France?",
    completion="Paris is the capital of France with 2.2 million people.",
    context_docs=context_docs
)
""")
    
    print("\n3. Custom Configuration:")
    print("""
from gpt_oss.monitoring import MonitorConfig, MonitorThresholds

config = MonitorConfig(
    k_samples=10,
    temperature=0.8,
    enable_retrieval_support=True,
    enable_jailbreak_heuristics=True,
    thresholds=MonitorThresholds(high=0.8, medium=0.5),
    html_report=True,
    report_dir="my_reports"
)

monitor = HallucinationMonitor(config)
""")

def show_detection_signals():
    """Show information about detection signals."""
    print("\n" + "="*60)
    print("DETECTION SIGNALS")
    print("="*60)
    
    signals = {
        "Self-Consistency (SC)": {
            "weight": "0.25",
            "description": "Generates k samples and computes semantic agreement",
            "method": "Cosine similarity of sentence embeddings"
        },
        "NLI Faithfulness (NLI)": {
            "weight": "0.35",
            "description": "Checks sentence-level entailment against prompt and context",
            "method": "Natural Language Inference model"
        },
        "Numeric Sanity (NS)": {
            "weight": "0.15",
            "description": "Detects arithmetic and unit consistency issues",
            "method": "Unit conversion and arithmetic checking"
        },
        "Retrieval Support (RS)": {
            "weight": "0.20",
            "description": "Verifies claims against provided context documents",
            "method": "Semantic similarity with context chunks"
        },
        "Jailbreak Heuristics (JB)": {
            "weight": "0.05",
            "description": "Identifies potential safety risks and jailbreak attempts",
            "method": "Pattern matching and keyword analysis"
        }
    }
    
    for signal_name, info in signals.items():
        print(f"\n{signal_name}:")
        print(f"  Weight: {info['weight']}")
        print(f"  Description: {info['description']}")
        print(f"  Method: {info['method']}")

def show_output_format():
    """Show the output format."""
    print("\n" + "="*60)
    print("OUTPUT FORMAT")
    print("="*60)
    
    example_output = {
        "risk_score": 0.65,
        "risk_level": "medium",
        "signals": {
            "self_consistency": 0.8,
            "nli_faithfulness": 0.6,
            "numeric_sanity": 0.9,
            "retrieval_support": 0.7,
            "jailbreak_heuristics": 0.1
        },
        "spans": [
            {
                "text": "contradictory sentence",
                "start": 100,
                "end": 120,
                "reason": "NLI: contradiction",
                "color": "red",
                "severity": "high"
            }
        ],
        "artifacts": {
            "html_report": "/path/to/report.html"
        },
        "config": {
            "k_samples": 5,
            "temperature": 0.7,
            "thresholds": {"high": 0.7, "medium": 0.4}
        },
        "timestamp": "2024-01-15T10:30:00",
        "prompt": "What is the capital of France?",
        "completion": "Paris is the capital of France with 2.2 million people."
    }
    
    print("\nExample JSON Output:")
    print(json.dumps(example_output, indent=2))

def show_features():
    """Show key features."""
    print("\n" + "="*60)
    print("KEY FEATURES")
    print("="*60)
    
    features = [
        "🔧 Configurable: Customize thresholds, weights, and detection parameters",
        "📊 HTML Reports: Beautiful, interactive reports with highlighted spans",
        "💻 CLI Interface: Easy command-line usage with file inputs",
        "⚡ Lightweight: CPU-optional with fallback heuristics",
        "🎯 Deterministic: Seeded RNG for reproducible results",
        "🛡️ Safety: Jailbreak detection and safety risk assessment",
        "📈 Multiple Signals: Five different detection methods",
        "🔍 Span Highlighting: Precise location of issues in text",
        "📝 Documentation: Comprehensive examples and guides"
    ]
    
    for feature in features:
        print(f"  {feature}")

def show_installation():
    """Show installation instructions."""
    print("\n" + "="*60)
    print("INSTALLATION")
    print("="*60)
    
    print("\n1. Install with monitoring dependencies:")
    print("   pip install -e '.[monitoring]'")
    
    print("\n2. Install CLI tool:")
    print("   pip install -e '.[monitoring]'")
    print("   # The 'gpt-oss-monitor' command will be available")
    
    print("\n3. Optional dependencies:")
    print("   - sentence-transformers: For semantic similarity")
    print("   - transformers: For NLI models")
    print("   - torch: For GPU acceleration")
    print("   - jinja2: For HTML report generation")

def main():
    """Run the demo."""
    print("🤖 GPT-OSS HALLUCINATION MONITOR DEMO")
    print("="*60)
    
    # Create demo files
    create_demo_files()
    
    # Show various aspects
    show_installation()
    show_features()
    show_detection_signals()
    show_cli_usage()
    show_python_api()
    show_output_format()
    
    print("\n" + "="*60)
    print("🎉 DEMO COMPLETE")
    print("="*60)
    print("\nTo get started:")
    print("1. Install dependencies: pip install -e '.[monitoring]'")
    print("2. Try the CLI: gpt-oss-monitor --prompt demo_prompt.txt --completion demo_completion.txt")
    print("3. Check the examples: gpt_oss/monitoring/examples/README.md")
    print("4. Read the design: docs/monitoring_design.md")

if __name__ == "__main__":
    main()
