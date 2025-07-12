import os
import threading
import time
from typing import Dict, Any

from rich.console import Console
from rich.table import Table
from rich.live import Live

from comind.core.agent.metric import MetricValue, WorstMetricValue

class Dashboard:
    def __init__(self):
        self.console = Console()
        self.agent_statuses: Dict[str, Any] = {}
        self.agent_last_seen: Dict[str, float] = {}
        self.lock = threading.Lock()
        
        self.agent_metrics: Dict[str, MetricValue] = {}
        self.global_best_metric: MetricValue = WorstMetricValue()

        self.live = Live(self.generate_table(), screen=True, redirect_stderr=False, refresh_per_second=2)
        self.connection_timeout = 60

    def generate_table(self) -> Table:
        title = "CoMind Community Dashboard"
        if not self.global_best_metric.is_worst:
            title += f" - Global Best Metric: {self.global_best_metric}"
        
        table = Table(title=title, caption="Agents")
        table.add_column("Agent ID", justify="left", style="cyan", no_wrap=True)
        table.add_column("Connection", justify="center", style="bold")
        table.add_column("Status", justify="left", style="magenta")
        table.add_column("Current Query/Task", justify="left", style="green", max_width=80)
        table.add_column("Agent Best Metric", justify="center", style="blue")
        table.add_column("Last Seen", justify="center", style="yellow")

        current_time = time.time()
        with self.lock:
            for agent_id, status_data in self.agent_statuses.items():
                state = status_data.get("state", "N/A")
                query = status_data.get("query", "")
                last_seen = self.agent_last_seen.get(agent_id, current_time)
                
                time_since_last_seen = current_time - last_seen
                connection_status = "[green]CONNECTED[/green]"
                if time_since_last_seen > self.connection_timeout:
                    connection_status = "[red]DISCONNECTED[/red]"
                    state = "[red]OFFLINE[/red]"
                
                if time_since_last_seen < 60:
                    last_seen_str = f"{int(time_since_last_seen)}s ago"
                elif time_since_last_seen < 3600:
                    last_seen_str = f"{int(time_since_last_seen/60)}m ago"
                else:
                    last_seen_str = f"{int(time_since_last_seen/3600)}h ago"
                
                display_query = query
                if len(query) > 80:
                    display_query = "..." + query[-77:]
                
                agent_metric_str = str(self.agent_metrics.get(agent_id, "N/A"))

                table.add_row(agent_id, connection_status, state, display_query, agent_metric_str, last_seen_str)
        return table

    def update_agent_status(self, agent_id: str, status: Dict[str, Any]):
        current_time = time.time()
        with self.lock:
            if agent_id not in self.agent_statuses:
                self.agent_statuses[agent_id] = {}
            self.agent_statuses[agent_id].update(status)
            self.agent_last_seen[agent_id] = current_time
        self.refresh()

    def update_metrics(self, agent_id: str, agent_metric: MetricValue, global_best_metric: MetricValue):
        """Update the metrics for a specific agent and the global best."""
        with self.lock:
            self.agent_metrics[agent_id] = agent_metric
            self.global_best_metric = global_best_metric
        self.refresh()

    def add_agent(self, agent_id: str):
        current_time = time.time()
        with self.lock:
            if agent_id not in self.agent_statuses:
                self.agent_statuses[agent_id] = {"state": "connected", "query": "Waiting for task..."}
            self.agent_last_seen[agent_id] = current_time
        self.refresh()
    
    def remove_agent(self, agent_id: str):
        """Remove an agent from the dashboard"""
        with self.lock:
            self.agent_statuses.pop(agent_id, None)
            self.agent_last_seen.pop(agent_id, None)
        self.refresh()
    
    def get_connected_agents(self) -> list:
        """Get list of currently connected agents"""
        current_time = time.time()
        connected_agents = []
        with self.lock:
            for agent_id, last_seen in self.agent_last_seen.items():
                if current_time - last_seen <= self.connection_timeout:
                    connected_agents.append(agent_id)
        return connected_agents

    def get_stale_agents(self, timeout: int) -> list[str]:
        """Returns a list of agent IDs that have not sent a heartbeat in `timeout` seconds."""
        stale_agents = []
        current_time = time.time()
        with self.lock:
            for agent_id, last_seen in self.agent_last_seen.items():
                if current_time - last_seen > timeout:
                    stale_agents.append(agent_id)
        return stale_agents

    def refresh(self):
        self.live.update(self.generate_table())

    def start(self):
        self.live.start()
        # Start background refresh thread
        self._running = True
        self._refresh_thread = threading.Thread(target=self._background_refresh, daemon=True)
        self._refresh_thread.start()

    def stop(self):
        self._running = False
        if hasattr(self, '_refresh_thread'):
            self._refresh_thread.join(timeout=1)
        self.live.stop()
        self.console.print("Dashboard stopped.")
    
    def _background_refresh(self):
        """Background thread to refresh dashboard periodically"""
        while self._running:
            time.sleep(5)  # Refresh every 5 seconds
            if self._running:
                self.refresh() 

    def _render(self):
        """Renders the dashboard table."""
        os.system('cls' if os.name == 'nt' else 'clear')