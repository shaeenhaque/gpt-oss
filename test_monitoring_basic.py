#!/usr/bin/env python3
"""
Basic test script for the GPT-OSS Hallucination Monitor.
This tests the module structure and basic functionality without heavy dependencies.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test basic imports."""
    print("Testing imports...")
    
    try:
        # Test config imports
        from gpt_oss.monitoring.config import MonitorConfig, MonitorThresholds
        print("✓ Config imports successful")
        
        # Test basic structure
        config = MonitorConfig()
        print(f"✓ Config created: k_samples={config.k_samples}")
        
        thresholds = MonitorThresholds()
        print(f"✓ Thresholds created: high={thresholds.high}, medium={thresholds.medium}")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config_functionality():
    """Test configuration functionality."""
    print("\nTesting configuration...")
    
    try:
        from gpt_oss.monitoring.config import MonitorConfig, MonitorThresholds
        
        # Test custom config
        config = MonitorConfig(
            k_samples=10,
            temperature=0.8,
            enable_retrieval_support=False
        )
        
        assert config.k_samples == 10
        assert config.temperature == 0.8
        assert config.enable_retrieval_support is False
        print("✓ Custom config creation successful")
        
        # Test to_dict
        config_dict = config.to_dict()
        assert 'k_samples' in config_dict
        assert 'temperature' in config_dict
        assert 'thresholds' in config_dict
        print("✓ Config serialization successful")
        
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        "gpt_oss/monitoring/__init__.py",
        "gpt_oss/monitoring/config.py",
        "gpt_oss/monitoring/halluci_monitor.py",
        "gpt_oss/monitoring/detectors/__init__.py",
        "gpt_oss/monitoring/detectors/self_consistency.py",
        "gpt_oss/monitoring/detectors/nli_faithfulness.py",
        "gpt_oss/monitoring/detectors/numeric_sanity.py",
        "gpt_oss/monitoring/detectors/retrieval_support.py",
        "gpt_oss/monitoring/detectors/jailbreak_heuristics.py",
        "gpt_oss/monitoring/highlight/__init__.py",
        "gpt_oss/monitoring/highlight/span_align.py",
        "gpt_oss/monitoring/highlight/html_report.py",
        "gpt_oss/monitoring/metrics/__init__.py",
        "gpt_oss/monitoring/metrics/scoring.py",
        "gpt_oss/monitoring/__main__.py",
        "gpt_oss/monitoring/requirements-monitor.txt",
        "gpt_oss/monitoring/examples/README.md",
        "gpt_oss/monitoring/examples/truthfulqa_mini.jsonl",
        "gpt_oss/monitoring/examples/fever_mini.jsonl",
        "tests/monitoring/test_monitor_basic.py",
        "tests/monitoring/test_numeric_sanity.py",
        "tests/monitoring/test_nli_faithfulness.py",
        "tests/monitoring/test_self_consistency.py",
        ".github/workflows/ci-monitoring.yml",
        "docs/monitoring_design.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path}")
    
    if missing_files:
        print(f"\n✗ Missing files: {missing_files}")
        return False
    else:
        print("\n✓ All required files exist")
        return True

def test_cli_help():
    """Test CLI help functionality."""
    print("\nTesting CLI help...")
    
    try:
        # Test that the main module can be imported
        from gpt_oss.monitoring import __main__
        print("✓ CLI module import successful")
        return True
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("GPT-OSS Hallucination Monitor - Basic Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_functionality,
        test_file_structure,
        test_cli_help
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
