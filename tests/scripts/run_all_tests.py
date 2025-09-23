#!/usr/bin/env python3
"""
Comprehensive VLN Pipeline Test Runner
=====================================

This script runs comprehensive tests of the VLN pipeline with different
video/instruction combinations and organizes all outputs in the tests directory.

Usage:
    cd <project_root>
    python tests/scripts/run_all_tests.py
    
    # Or with custom config:
    python tests/scripts/run_all_tests.py --config tests/configs/test_config.yaml
"""

import torch
import yaml
import sys
import argparse
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def load_test_config(config_path: str) -> Dict[str, Any]:
    """Load test configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_test_session_dir() -> Path:
    """Create a timestamped test session directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = project_root / "tests" / "results" / f"test_session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir

def run_single_test(
    video_path: str, 
    instruction_path: str, 
    output_dir: str,
    test_name: str
) -> Dict[str, Any]:
    """Run a single test and return results."""
    
    print(f"\n{'='*60}")
    print(f"Running Test: {test_name}")
    print(f"Video: {video_path}")
    print(f"Instruction: {instruction_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    # Prepare command
    test_script = project_root / "tests" / "scripts" / "test_pipeline.py"
    
    cmd = [
        "python", str(test_script),
        "--video", video_path,
        "--output_dir", output_dir
    ]
    
    # Add instruction (file or text)
    if instruction_path.endswith('.txt') or instruction_path.endswith('.json'):
        cmd.extend(["--instruction_file", instruction_path])
    else:
        cmd.extend(["--instruction", instruction_path])
    
    start_time = time.time()
    
    try:
        # Run the test
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300,
            cwd=str(project_root)
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Parse results
        test_result = {
            "test_name": test_name,
            "video_path": video_path,
            "instruction_path": instruction_path,
            "output_dir": output_dir,
            "duration_seconds": duration,
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
        
        if result.returncode == 0:
            print(f"✅ SUCCESS - {test_name} ({duration:.1f}s)")
        else:
            print(f"❌ FAILED - {test_name} ({duration:.1f}s)")
            print(f"Error: {result.stderr}")
            
        return test_result
        
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT - {test_name}")
        return {
            "test_name": test_name,
            "success": False,
            "error": "Test timed out after 300 seconds",
            "duration_seconds": 300
        }
    except Exception as e:
        print(f"💥 ERROR - {test_name}: {str(e)}")
        return {
            "test_name": test_name,
            "success": False,
            "error": str(e),
            "duration_seconds": 0
        }

def run_comprehensive_tests(config_path: str) -> None:
    """Run comprehensive test suite."""
    
    print("🚀 Starting VLN Pipeline Comprehensive Test Suite")
    print(f"Config: {config_path}")
    print(f"Project Root: {project_root}")
    
    # Load configuration
    config = load_test_config(config_path)
    
    # Create test session directory
    session_dir = create_test_session_dir()
    print(f"Test Session Directory: {session_dir}")
    
    # Prepare test combinations
    test_cases = []
    
    for video_name, video_path in config['test_videos'].items():
        full_video_path = f"../{video_path}"
        
        for instr_name, instr_path in config['test_instructions'].items():
            full_instr_path = str(project_root / "tests" / "instructions" / instr_path)
            
            test_name = f"{video_name}_{instr_name}"
            output_dir = str(session_dir / test_name)
            
            test_cases.append({
                "name": test_name,
                "video_path": full_video_path,
                "instruction_path": full_instr_path,
                "output_dir": output_dir
            })
    
    print(f"\n📋 Test Plan: {len(test_cases)} tests to run")
    for i, test in enumerate(test_cases, 1):
        print(f"  {i:2d}. {test['name']}")
    
    # Run all tests
    results = []
    start_time = time.time()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}/{len(test_cases)}")
        
        result = run_single_test(
            test_case['video_path'],
            test_case['instruction_path'],
            test_case['output_dir'],
            test_case['name']
        )
        
        results.append(result)
    
    total_time = time.time() - start_time
    
    # Generate summary report
    successful_tests = [r for r in results if r.get('success', False)]
    failed_tests = [r for r in results if not r.get('success', False)]
    
    print(f"\n{'='*60}")
    print("🎯 TEST SUITE SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {len(test_cases)}")
    print(f"Successful: {len(successful_tests)} ✅")
    print(f"Failed: {len(failed_tests)} ❌")
    print(f"Success Rate: {len(successful_tests)/len(test_cases)*100:.1f}%")
    print(f"Total Duration: {total_time:.1f} seconds")
    
    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for test in failed_tests:
            print(f"  - {test['test_name']}: {test.get('error', 'Unknown error')}")
    
    # Save detailed results
    results_file = session_dir / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "test_session": str(session_dir),
            "config_used": config,
            "total_tests": len(test_cases),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "success_rate": len(successful_tests)/len(test_cases),
            "total_duration_seconds": total_time,
            "results": results
        }, f, indent=2)
    
    print(f"\n📊 Detailed results saved to: {results_file}")
    print(f"📁 All test outputs in: {session_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive VLN pipeline tests")
    parser.add_argument(
        "--config", 
        type=str, 
        default=str(project_root / "tests" / "configs" / "test_config.yaml"),
        help="Path to test configuration file"
    )
    
    args = parser.parse_args()
    
    if not Path(args.config).exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    run_comprehensive_tests(args.config)

if __name__ == "__main__":
    main()