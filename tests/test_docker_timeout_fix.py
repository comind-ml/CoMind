#!/usr/bin/env python3
"""
Test script to verify Docker timeout fixes
"""

import sys
import asyncio
import tempfile
import os
from pathlib import Path

from comind.core.config import DockerConfig
from comind.environment.docker import Docker

async def test_docker_timeout_fix():
    print("Testing Docker timeout fixes...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Docker configuration
        docker_config = DockerConfig(
            image="python:3.11-slim",
            mounts=[f"{temp_path}:/workspace"],
            code_path="/workspace/main.py",
            timeout=10  # Enough time for the 3-second test
        )
        
        docker_manager = Docker(docker_config)
        
        print("✓ Docker manager created")
        
        # Test 1: Quick execution that should succeed
        print("\n--- Test 1: Quick execution ---")
        quick_code = """
print("Hello, World!")
print("This is a quick test")
import sys
sys.exit(0)
"""
        
        try:
            exit_code = await docker_manager.run_program(quick_code)
            logs = await docker_manager.get_logs()
            print(f"Quick execution - Exit code: {exit_code}")
            print(f"Logs length: {len(logs)} chars")
            print(f"Logs preview: {logs[:200]}...")
        except Exception as e:
            print(f"Quick execution failed: {e}")
        
        # Test 2: Code that will error and exit quickly
        print("\n--- Test 2: Error execution ---")
        error_code = """
print("Starting error test")
raise ValueError("This is a test error")
"""
        
        try:
            exit_code = await docker_manager.run_program(error_code)
            logs = await docker_manager.get_logs()
            print(f"Error execution - Exit code: {exit_code}")
            print(f"Logs length: {len(logs)} chars")
            print(f"Logs preview: {logs[:200]}...")
        except Exception as e:
            print(f"Error execution failed: {e}")
        
        # Test 3: Code that takes some time but finishes
        print("\n--- Test 3: Timed execution ---")
        timed_code = """
import time
print("Starting timed test")
for i in range(1000):
    print(f"Step {i+1}")
    time.sleep(1)
print("Timed test completed")
"""
        
        try:
            exit_code = await docker_manager.run_program(timed_code)
            logs = await docker_manager.get_logs()
            print(f"Timed execution - Exit code: {exit_code}")
            print(f"Logs length: {len(logs)} chars")
            print(f"Logs preview: {logs[:200]}...")
        except Exception as e:
            print(f"Timed execution failed: {e}")
        
        # Clean up
        docker_manager.stop_container()
        print("\n✓ Docker timeout fixes test completed!")

if __name__ == "__main__":
    asyncio.run(test_docker_timeout_fix()) 