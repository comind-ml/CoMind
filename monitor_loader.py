#!/usr/bin/env python3
"""
Simple script to load and monitor a saved agent state.
Usage: python monitor_loader.py <path_to_agent_state.pkl>
"""

import sys
import pickle
import time
import threading
from pathlib import Path
from comind.monitor import run_monitor

class AgentStateLoader:
    """Wrapper class to load agent state for monitoring with auto-refresh."""
    
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.last_modified = 0
        self._lock = threading.Lock()
        
        # Initialize workspace scanning for coders
        self.workspace_path = None
        self._last_coder_scan = 0
        
        # Initialize empty code_agents
        self.code_agents = []
        
        # Initialize global best metric tracking
        self.global_best_metric = "N/A"
        self._last_global_best = None
        
        # Load initial state
        self._load_state()
        
        # Immediately scan for coders
        self._scan_and_update_coders()
        
        # Start background thread to monitor file changes
        self._monitor_thread = threading.Thread(target=self._monitor_file_changes, daemon=True)
        self._monitor_thread.start()
    
    def _load_state(self):
        """Load state from pickle file."""
        try:
            with open(self.state_file, 'rb') as f:
                state = pickle.load(f)
            
            with self._lock:
                # Store old values for comparison
                old_ideas_count = len(getattr(self, 'ideas', []))
                old_reports_count = len(getattr(self, 'reports', []))
                old_datasets_count = len(getattr(self, 'datasets', {}))
                
                # Copy state attributes
                self.ideas = state.get('ideas', [])
                self.reports = state.get('reports', [])
                self.datasets = state.get('datasets', {})
                self.code_agents = state.get('code_agents', [])
                self.cfg = state.get('cfg')
                
                # Load global best metric
                old_global_best = self.global_best_metric
                self.global_best_metric = state.get('global_best_metric', 'N/A')
                
                if old_global_best != self.global_best_metric:
                    print(f"🏆 Global best metric updated: {old_global_best} -> {self.global_best_metric}")
                
                # Also update code_agents from main state if available
                if 'code_agents' in state and state['code_agents']:
                    # Merge with discovered coders, preferring main state data
                    main_code_agents = state['code_agents']
                    print(f"📊 Found {len(main_code_agents)} code agents in main state")
                
                # Check for changes
                ideas_changed = len(self.ideas) != old_ideas_count
                reports_changed = len(self.reports) != old_reports_count
                datasets_changed = len(self.datasets) != old_datasets_count
                
                if ideas_changed:
                    print(f"🔄 Ideas changed: {old_ideas_count} -> {len(self.ideas)}")
                if reports_changed:
                    print(f"🔄 Reports changed: {old_reports_count} -> {len(self.reports)}")
                if datasets_changed:
                    print(f"🔄 Datasets changed: {old_datasets_count} -> {len(self.datasets)}")
                
                # Set workspace path for coder scanning
                if self.cfg and hasattr(self.cfg, 'agent_workspace_dir'):
                    self.workspace_path = Path(self.cfg.agent_workspace_dir)
                    print(f"📊 Workspace path: {self.workspace_path}")
                
                # Set a flag to indicate state has been updated
                self._state_updated = True
                
            # Update last modified time
            self.last_modified = self.state_file.stat().st_mtime
            print(f"📊 State reloaded: {len(self.reports)} reports, {len(self.ideas)} ideas, {len(self.datasets)} datasets")
            
        except Exception as e:
            print(f"Error loading state: {e}")
    
    def has_state_updated(self):
        """Check if state has been updated since last check."""
        with self._lock:
            if hasattr(self, '_state_updated') and self._state_updated:
                self._state_updated = False
                return True
            return False
    
    def _scan_coder_states(self):
        """Scan workspace for coder state files."""
        discovered_coders = []
        
        if not self.workspace_path or not self.workspace_path.exists():
            return discovered_coders
            
        try:
            # Look for coder_state.pkl files in subdirectories
            for subdir in self.workspace_path.glob("*/"):
                print(f"📊 Scanning for coders in {subdir}")
                if subdir.is_dir():
                    coder_state_file = subdir / "coder_state.pkl"
                    if coder_state_file.exists():
                        try:
                            with open(coder_state_file, 'rb') as f:
                                coder_state = pickle.load(f)
                            
                            # Convert to expected format
                            messages = coder_state.get('messages', [])
                            
                            output_lines = coder_state.get('output_lines', [])
                            print(f"📊 Coder {subdir.name}: {len(output_lines)} output lines")
                            
                            # Get best metric and format it properly
                            best_metric = coder_state.get('best_metric', 'N/A')
                            is_running = coder_state.get('is_running', False)
                            completed = coder_state.get('completed', False)
                            
                            # Status indicator
                            status = "🔄 Running" if is_running else ("✅ Completed" if completed else "⏸️ Stopped")
                            
                            coder_data = {
                                "name": coder_state.get('name', f'Coder {len(discovered_coders)+1}'),
                                "messages": messages,
                                "code": coder_state.get('code', ''),
                                "output_lines": output_lines,
                                "iteration": coder_state.get('iteration', 0),
                                "best_metric": best_metric,
                                "draft_id": coder_state.get('draft_id', subdir.name),
                                "is_running": is_running,
                                "completed": completed,
                                "status": status
                            }
                            discovered_coders.append(coder_data)
                            
                        except Exception as e:
                            print(f"Error loading coder state from {coder_state_file}: {e}")
        except Exception as e:
            print(f"Error scanning coder states: {e}")
        
        return discovered_coders
    
    def _scan_and_update_coders(self):
        """Scan for coders and update the code_agents list."""
        try:
            discovered_coders = self._scan_coder_states()
            with self._lock:
                old_count = len(self.code_agents)
                # Only update state flag if there's an actual change
                if len(discovered_coders) != old_count:
                    self.code_agents = discovered_coders
                    self._state_updated = True
                    print(f"🔄 Updated coders: {len(discovered_coders)} (was {old_count})")
                else:
                    # Update the data but don't trigger UI refresh
                    self.code_agents = discovered_coders
                
                # Update global best metric from all coders
                self._update_global_best_metric()
        except Exception as e:
            print(f"Error scanning coders: {e}")
    
    def _update_global_best_metric(self):
        """Update global best metric from all code agents."""
        try:
            best_metrics = []
            
            # Collect all valid best metrics from code agents
            for agent_data in self.code_agents:
                best_metric = agent_data.get("best_metric")
                if best_metric and best_metric != "N/A" and best_metric != "None" and best_metric != "WorstMetricValue()":
                    try:
                        # Handle different metric string formats
                        metric_str = str(best_metric).strip()
                        
                        # Remove common prefixes/suffixes from metric strings
                        if metric_str.startswith("MetricValue(") and metric_str.endswith(")"):
                            # Extract number from MetricValue(0.123456)
                            metric_str = metric_str[12:-1]
                        
                        # Try to convert to float for comparison
                        metric_value = float(metric_str)
                        best_metrics.append(metric_value)
                        print(f"📊 Valid metric from {agent_data.get('draft_id', 'unknown')}: {metric_value}")
                    except (ValueError, TypeError):
                        # If not a number, keep as string for display
                        print(f"📊 Non-numeric metric from {agent_data.get('draft_id', 'unknown')}: {best_metric}")
                        best_metrics.append(best_metric)
            
            if best_metrics:
                # If all metrics are numbers, find the best (assuming higher is better)
                if all(isinstance(m, (int, float)) for m in best_metrics):
                    new_global_best = max(best_metrics)
                    # Format to reasonable precision
                    if isinstance(new_global_best, float):
                        new_global_best = f"{new_global_best:.6f}"
                    else:
                        new_global_best = str(new_global_best)
                else:
                    # Mixed types, just show the first valid one
                    new_global_best = str(best_metrics[0])
                
                old_global_best = self.global_best_metric
                # Only update if there's a meaningful change
                if new_global_best != old_global_best and old_global_best != "N/A":
                    self.global_best_metric = new_global_best
                    self._state_updated = True
                    print(f"🏆 Global best metric updated from coders: {old_global_best} -> {new_global_best}")
                elif old_global_best == "N/A":
                    # First time setting a valid metric
                    self.global_best_metric = new_global_best
                    self._state_updated = True
                    print(f"🏆 Global best metric initialized from coders: {new_global_best}")
            else:
                # No valid metrics found from coders, but don't override if we have one from main state
                print("📊 No valid metrics found from individual coders")
                    
        except Exception as e:
            print(f"Error updating global best metric: {e}")
    
    def _monitor_file_changes(self):
        """Monitor file changes and reload when needed."""
        while True:
            try:
                # Check main state file
                if self.state_file.exists():
                    current_modified = self.state_file.stat().st_mtime
                    if current_modified > self.last_modified:
                        print("🔄 State file updated, reloading...")
                        self._load_state()
                
                # Check for coder states every 3 seconds
                current_time = time.time()
                if current_time - self._last_coder_scan > 3:
                    self._scan_and_update_coders()
                    self._last_coder_scan = current_time
                
                time.sleep(2)  # Check every 2 seconds
            except Exception as e:
                print(f"Error monitoring file: {e}")
                time.sleep(5)

def main():
    if len(sys.argv) != 2:
        print("Usage: python monitor_loader.py <path_to_agent_state.pkl>")
        sys.exit(1)
    
    state_file = Path(sys.argv[1])
    if not state_file.exists():
        print(f"Error: State file {state_file} not found")
        sys.exit(1)
    
    try:
        agent = AgentStateLoader(state_file)
        print(f"📊 Loading agent state from {state_file}")
        print(f"📈 Found {len(agent.reports)} reports, {len(agent.ideas)} ideas, {len(agent.datasets)} datasets")
        run_monitor(agent)
    except Exception as e:
        print(f"Error loading agent state: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
