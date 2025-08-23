"""
Fancy monitoring panel for the agent's running states using textual.
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    TabbedContent, TabPane, Static, ListView, ListItem, 
    Markdown, Label, Tree, DirectoryTree, Select
)
from textual.binding import Binding
from textual.message import Message
from textual.reactive import reactive
from textual.css.query import NoMatches
from textual import events
from pathlib import Path
from rich.syntax import Syntax
from rich.console import Console
from rich.text import Text
from rich.markdown import Markdown as RichMarkdown
import os
import re

from comind.community import Pipeline, Dataset
from comind.agent import Agent


def clean_coder_name(name: str) -> str:
    """Clean coder name by removing dots and keeping only letters, numbers, and hyphens."""
    if not name:
        return name
    # Remove dots and keep only letters, numbers, hyphens, and spaces
    cleaned = re.sub(r'[^a-zA-Z0-9\-\s]', '', str(name))
    # Replace multiple spaces with single space and strip
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def sanitize_widget_id(id_str: str) -> str:
    """Sanitize string to be a valid Textual widget ID."""
    if not id_str:
        return id_str
    # Replace invalid characters with hyphens
    # Textual IDs can only contain letters, numbers, underscores, and hyphens
    sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '-', str(id_str))
    # Remove consecutive hyphens
    sanitized = re.sub(r'-+', '-', sanitized)
    # Remove leading/trailing hyphens
    sanitized = sanitized.strip('-')
    return sanitized


def normalize_markdown(markdown_text: str) -> str:
    """Normalize markdown to improve list rendering.

    - Convert CRLF to LF
    - Dedent list items mistakenly indented >=4 spaces (avoids code blocks)
    - Ensure a blank line before list starts so the parser recognizes lists
    """
    if not markdown_text:
        return ""

    text = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    list_start_pattern = re.compile(r"^(?:\d+\.\s|[-*+]\s)")
    out_lines: list[str] = []
    previous_was_blank = True

    for raw_line in lines:
        line = raw_line
        stripped = line.lstrip(" ")
        leading_spaces = len(line) - len(stripped)

        # If a potential list item is indented >= 4 spaces, dedent to avoid being treated as code block
        if leading_spaces >= 4 and list_start_pattern.match(stripped):
            line = stripped

        # Ensure a blank line before a list start if previous wasn't blank
        if not previous_was_blank and list_start_pattern.match(line):
            out_lines.append("")

        out_lines.append(line)
        previous_was_blank = (line.strip() == "")

    return "\n".join(out_lines)

class CodeViewer(Static):
    """Widget for displaying syntax-highlighted Python code."""
    
    def __init__(self, code: str = "", **kwargs):
        super().__init__(**kwargs)
        self.code = code
    
    def on_mount(self) -> None:
        if self.code:
            console = Console()
            syntax = Syntax(self.code, "python", theme="monokai", line_numbers=True)
            self.update(syntax)


class MarkdownViewer(Static):
    """Widget for displaying Markdown via Rich renderer."""
    
    def __init__(self, text: str = "", **kwargs):
        super().__init__(**kwargs)
        self.text = text
    
    def on_mount(self) -> None:
        if self.text is not None:
            # Escape HTML tags to show them as literal text
            markdown = RichMarkdown(self.text, hyperlinks=False, code_theme="monokai")
            self.update(markdown)


class OutputViewer(Static):
    """Widget for displaying plain text output with scrolling parent."""

    def __init__(self, lines=None, **kwargs):
        super().__init__(**kwargs)
        self._lines = [str(x) for x in (lines or [])]

    def on_mount(self) -> None:
        self._render_lines()

    def set_lines(self, lines) -> None:
        self._lines = [str(x) for x in (lines or [])]
        self._render_lines()

    def _render_lines(self) -> None:
        text = "\n".join(self._lines)
        # Render output in a code-like style
        self.update(Syntax(text, "text", theme="monokai", line_numbers=False))


class MessageBubble(Container):
    """A single chat bubble with role-based styling."""

    def __init__(self, role: str, content: str, **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.content = content
        self.add_class("message-bubble")
        self.add_class(f"role-{role}")

    def compose(self) -> ComposeResult:
        yield MarkdownViewer(self.content)


class MessageRow(Horizontal):
    """A row containing a bubble aligned left (llm) or right (agent)."""

    def __init__(self, role: str, content: str, **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.content = content
        self.add_class("message-row")
        self.add_class(f"role-{role}")

    def compose(self) -> ComposeResult:
        if self.role == "agent":
            yield Container(classes="spacer")
            yield MessageBubble(self.role, self.content)
        else:
            yield MessageBubble(self.role, self.content)
            yield Container(classes="spacer")


class RoundedContainer(Container):
    """Container with rounded corners styling."""
    
    DEFAULT_CSS = """
    RoundedContainer {
        border: round $accent;
        background: $surface;
    }
    """


class ReportDetailView(Container):
    """Detailed view for a single report."""
    
    def __init__(self, report: Pipeline, **kwargs):
        super().__init__(**kwargs)
        self.report = report
    
    def compose(self) -> ComposeResult:
        with Vertical():
            # Upper half: Description in markdown
            with VerticalScroll(id="description-container"):
                yield MarkdownViewer(normalize_markdown(self.report.description), id="report-description")
            
            # Lower half: Code with syntax highlighting
            with VerticalScroll(id="code-container"):
                yield CodeViewer(self.report.code, id="report-code")
    
    DEFAULT_CSS = """
    ReportDetailView {
        height: 100%;
        width: 100%;
    }
    
    #report-code {
        height: auto;
        scrollbar-gutter: stable;
    }
    """


class DatasetDetailView(Container):
    """Detailed view for a single dataset."""
    
    def __init__(self, dataset: Dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
    
    def compose(self) -> ComposeResult:
        with Vertical():
            # Upper half: Description in markdown
            with VerticalScroll(id="dataset-description-container"):
                yield MarkdownViewer(normalize_markdown(self.dataset.description), id="dataset-description")
            
            # Lower half: File tree
            with VerticalScroll(id="file-tree-container"):
                if self.dataset.base_path.exists():
                    yield DirectoryTree(str(self.dataset.base_path), id="dataset-tree")
                else:
                    yield Label("Dataset path not found", id="dataset-error")
    
    DEFAULT_CSS = """
    DatasetDetailView {
        height: 100%;
        width: 100%;
    }
    
    #dataset-tree {
        height: 100%;
        scrollbar-gutter: stable;
    }
    """


class IdeaItem(Container):
    """Individual idea item with rounded background."""
    
    def __init__(self, idea: str, **kwargs):
        super().__init__(**kwargs)
        self.idea = idea
    
    def compose(self) -> ComposeResult:
        # Use MarkdownViewer to ensure HTML tags are escaped
        yield MarkdownViewer(self.idea)
    
    DEFAULT_CSS = """
    IdeaItem {
        height: auto;
        padding: 0;
    }
    """


class LeftPanel(RoundedContainer):
    """Left panel containing the three tabs: Reports, Ideas, Datasets."""
    
    def __init__(self, agent: Agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self.current_view = "list"  # "list" or "detail"
        self.current_report = None
        self.current_dataset = None
        self._refresh_timer = None
    
    def compose(self) -> ComposeResult:
        with TabbedContent(initial="reports"):
            # Reports Tab
            with TabPane("Reports", id="reports"):
                report_items = []
                for i, report in enumerate(self.agent.reports):
                    # Sanitize ID to avoid invalid characters
                    safe_id = f"report-{i}-{sanitize_widget_id(report.id)}"
                    item = ListItem(Label(report.title), id=safe_id)
                    item.report = report
                    report_items.append(item)
                yield ListView(*report_items, id="reports-list")
            
            # Ideas Tab  
            with TabPane("Ideas", id="ideas"):
                with VerticalScroll(id="ideas-scroll"):
                    for i, idea in enumerate(self.agent.ideas):
                        yield IdeaItem(idea, id=f"idea-{i}")
            
            # Datasets Tab
            with TabPane("Datasets", id="datasets"):
                dataset_items = []
                for i, (dataset_id, dataset) in enumerate(self.agent.datasets.items()):
                    # Sanitize ID to avoid invalid characters
                    safe_id = f"dataset-{i}-{sanitize_widget_id(dataset_id)}"
                    item = ListItem(Label(dataset.name), id=safe_id)
                    item.dataset = dataset
                    dataset_items.append(item)
                yield ListView(*dataset_items, id="datasets-list")
    
    def on_mount(self) -> None:
        # Debug message to verify data presence
        print(f"LeftPanel mounted with {len(self.agent.reports)} reports, {len(self.agent.ideas)} ideas, {len(self.agent.datasets)} datasets")
        self.call_later(self._focus_current_tab)
        
        # Start periodic refresh to update content when agent state changes
        try:
            self._refresh_timer = self.set_interval(3.0, self._refresh_content)
        except Exception:
            pass
    
    def on_unmount(self) -> None:
        # Stop periodic refresh when panel is removed
        try:
            if self._refresh_timer is not None:
                self._refresh_timer.stop()
        except Exception:
            pass
    
    def _refresh_content(self) -> None:
        """Refresh the content to reflect any changes in agent state."""
        try:
            # Check if agent state has been updated (for monitor_loader)
            state_updated = False
            if hasattr(self.agent, 'has_state_updated') and callable(self.agent.has_state_updated):
                state_updated = self.agent.has_state_updated()
            
            # Always try to refresh if we're in list view, let each method decide based on actual changes
            if self.current_view == "list":
                tabbed_content = self.query_one(TabbedContent)
                active_tab = tabbed_content.active
                
                # Check all tabs for updates, not just the active one
                self._refresh_reports_list(False)
                self._refresh_ideas_list(False)
                self._refresh_datasets_list(False)
                
                # If state was updated, print a debug message
                if state_updated:
                    print(f"🔄 Content refreshed for active tab: {active_tab}")
        except Exception as e:
            print(f"Error in _refresh_content: {e}")
            pass
    
    def _refresh_reports_list(self, force_refresh=False):
        """Refresh the reports list."""
        try:
            reports_list = self.query_one("#reports-list", ListView)
            current_count = len(reports_list.children)
            new_count = len(self.agent.reports)
            
            # Debug: Print current state
            print(f"🔍 Reports check: current_count={current_count}, new_count={new_count}, agent.reports type={type(self.agent.reports)}")
            
            # Only refresh if there's actually a change in count, not on force_refresh
            if current_count != new_count:
                reports_pane = self.query_one("#reports", TabPane)
                reports_pane.remove_children()
                report_items = []
                for i, report in enumerate(self.agent.reports):
                    safe_id = f"report-{i}-{sanitize_widget_id(report.id)}"
                    item = ListItem(Label(report.title), id=safe_id)
                    item.report = report
                    report_items.append(item)
                reports_pane.mount(ListView(*report_items, id="reports-list"))
                print(f"🔄 Reports refreshed: {new_count} items (was {current_count})")
        except Exception as e:
            print(f"Error refreshing reports: {e}")
            pass
    
    def _refresh_ideas_list(self, force_refresh=False):
        """Refresh the ideas list."""
        try:
            ideas_scroll = self.query_one("#ideas-scroll", VerticalScroll)
            current_count = len(ideas_scroll.children)
            new_count = len(self.agent.ideas)
            
            # Debug: Print current state
            print(f"🔍 Ideas check: current_count={current_count}, new_count={new_count}, agent.ideas type={type(self.agent.ideas)}")
            if len(self.agent.ideas) > 0:
                print(f"🔍 First few ideas: {self.agent.ideas[:2]}")
            
            # Only refresh if there's actually a change in count
            if current_count != new_count:
                ideas_pane = self.query_one("#ideas", TabPane)
                ideas_pane.remove_children()
                
                # Create new VerticalScroll with idea items as children
                idea_items = []
                for i, idea in enumerate(self.agent.ideas):
                    idea_items.append(IdeaItem(idea, id=f"idea-{i}"))
                
                new_scroll = VerticalScroll(*idea_items, id="ideas-scroll")
                ideas_pane.mount(new_scroll)
                print(f"🔄 Ideas refreshed: {new_count} items (was {current_count})")
        except Exception as e:
            print(f"Error refreshing ideas: {e}")
            pass
    
    def _refresh_datasets_list(self, force_refresh=False):
        """Refresh the datasets list."""
        try:
            datasets_list = self.query_one("#datasets-list", ListView)
            current_count = len(datasets_list.children)
            new_count = len(self.agent.datasets)
            
            # Debug: Print current state
            print(f"🔍 Datasets check: current_count={current_count}, new_count={new_count}, agent.datasets type={type(self.agent.datasets)}")
            
            # Only refresh if there's actually a change in count
            if current_count != new_count:
                datasets_pane = self.query_one("#datasets", TabPane)
                datasets_pane.remove_children()
                dataset_items = []
                for i, (dataset_id, dataset) in enumerate(self.agent.datasets.items()):
                    safe_id = f"dataset-{i}-{sanitize_widget_id(dataset_id)}"
                    item = ListItem(Label(dataset.name), id=safe_id)
                    item.dataset = dataset
                    dataset_items.append(item)
                datasets_pane.mount(ListView(*dataset_items, id="datasets-list"))
                print(f"🔄 Datasets refreshed: {new_count} items (was {current_count})")
        except Exception as e:
            print(f"Error refreshing datasets: {e}")
            pass
    
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle list item selection."""
        item = event.item
        
        # Check which tab we're in
        tabbed_content = self.query_one(TabbedContent)
        active_tab = tabbed_content.active
        
        if active_tab == "reports" and hasattr(item, 'report'):
            self._show_report_detail(item.report)
        elif active_tab == "datasets" and hasattr(item, 'dataset'):
            self._show_dataset_detail(item.dataset)
    
    def _show_report_detail(self, report: Pipeline) -> None:
        """Show detailed view of a report."""
        print(f"Showing report detail: {report.title}")
        self.current_report = report
        self.current_view = "detail"
        
        # Replace the reports tab content with detail view
        reports_pane = self.query_one("#reports", TabPane)
        reports_pane.remove_children()
        reports_pane.mount(ReportDetailView(report))
        print("Report detail view created")
    
    def _show_dataset_detail(self, dataset: Dataset) -> None:
        """Show detailed view of a dataset."""
        print(f"Showing dataset detail: {dataset.name}")
        self.current_dataset = dataset
        self.current_view = "detail"
        
        # Replace the datasets tab content with detail view
        datasets_pane = self.query_one("#datasets", TabPane)
        datasets_pane.remove_children()
        datasets_pane.mount(DatasetDetailView(dataset))
        print("Dataset detail view created")
    
    def _focus_current_tab(self) -> None:
        tabbed = self.query_one(TabbedContent)
        active_tab_id = tabbed.active
        
        try:
            if active_tab_id == "ideas":
                self.query_one("#ideas-scroll").focus()
            
            elif active_tab_id == "reports":
                if self.current_view == "detail":
                    self.query_one(ReportDetailView).focus()
                else:
                    self.query_one("#reports-list").focus()

            elif active_tab_id == "datasets":
                if self.current_view == "detail":
                    self.query_one(DatasetDetailView).focus()
                else:
                    self.query_one("#datasets-list").focus()
        except NoMatches:
            # If the target widget doesn't exist (e.g., in detail view), do nothing.
            pass

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        # 布局切换后再聚焦，避免找不到 widget
        self.call_later(self._focus_current_tab)
    
    def _return_to_list(self) -> None:
        """Return to list view from detail view."""
        if self.current_view == "detail":
            tabbed_content = self.query_one(TabbedContent)
            active_tab = tabbed_content.active
            
            if active_tab == "reports" and self.current_report:
                # Restore reports list
                reports_pane = self.query_one("#reports", TabPane)
                reports_pane.remove_children()
                report_items = []
                for i, report in enumerate(self.agent.reports):
                    # Sanitize ID to avoid invalid characters
                    safe_id = f"report-{i}-{sanitize_widget_id(report.id)}"
                    item = ListItem(Label(report.title), id=safe_id)
                    item.report = report
                    report_items.append(item)
                reports_pane.mount(ListView(*report_items, id="reports-list"))
                self.current_report = None
                
            elif active_tab == "datasets" and self.current_dataset:
                # Restore datasets list
                datasets_pane = self.query_one("#datasets", TabPane)
                datasets_pane.remove_children()
                dataset_items = []
                for i, (dataset_id, dataset) in enumerate(self.agent.datasets.items()):
                    # Sanitize ID to avoid invalid characters
                    safe_id = f"dataset-{i}-{sanitize_widget_id(dataset_id)}"
                    item = ListItem(Label(dataset.name), id=safe_id)
                    item.dataset = dataset
                    dataset_items.append(item)
                datasets_pane.mount(ListView(*dataset_items, id="datasets-list"))
                self.current_dataset = None
            
            self.current_view = "list"
    
    def on_key(self, event: events.Key) -> None:
        """Handle key presses."""
        if event.key == "q" and self.current_view == "detail":
            self._return_to_list()
            event.prevent_default()


class MetricsPanel(RoundedContainer):
    """Panel for displaying best metrics information."""
    
    def __init__(self, agent: Agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self._metrics_refresh_timer = None
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("📊 Best Metrics", id="metrics-title")
            with VerticalScroll(id="metrics-scroll"):
                yield Static("Loading metrics...", id="metrics-content")
    
    def on_mount(self) -> None:
        self._refresh_metrics()
        # Refresh metrics every 2 seconds
        try:
            self._metrics_refresh_timer = self.set_interval(2.0, self._refresh_metrics)
        except Exception:
            pass
    
    def on_unmount(self) -> None:
        try:
            if self._metrics_refresh_timer is not None:
                self._metrics_refresh_timer.stop()
        except Exception:
            pass
    
    def _refresh_metrics(self) -> None:
        """Refresh the metrics display."""
        try:
            metrics_content = self.query_one("#metrics-content", Static)
            
            # Get global best metric
            global_best = getattr(self.agent, 'global_best_metric', 'N/A')
            
            # Get code agents metrics
            code_agents = getattr(self.agent, "code_agents", [])
            
            # Build metrics text
            lines = []
            lines.append(f"🏆 Global Best: {global_best}")
            lines.append("")
            
            if code_agents:
                lines.append("📈 Individual Agents:")
                for i, agent_data in enumerate(code_agents):
                    raw_name = agent_data.get("draft_id") or agent_data.get("name") or f"Agent {i+1}"
                    name = clean_coder_name(raw_name)
                    best_metric = agent_data.get("best_metric", "N/A")
                    iteration = agent_data.get("iteration", 0)
                    status = agent_data.get("status", "")
                    
                    # Format metric for display
                    if best_metric and best_metric != "N/A" and best_metric != "None":
                        # Handle MetricValue format
                        metric_str = str(best_metric)
                        if metric_str.startswith("MetricValue(") and metric_str.endswith(")"):
                            metric_str = metric_str[12:-1]
                        try:
                            metric_val = float(metric_str)
                            metric_display = f"{metric_val:.6f}"
                        except:
                            metric_display = metric_str
                    else:
                        metric_display = "N/A"
                    
                    lines.append(f"  {name}: {metric_display} (iter {iteration}) {status}")
            else:
                lines.append("No agents found")
            
            metrics_text = "\n".join(lines)
            
            # Use Syntax for better formatting
            syntax = Syntax(metrics_text, "text", theme="monokai", line_numbers=False)
            metrics_content.update(syntax)
            
        except Exception as e:
            print(f"Error refreshing metrics: {e}")
            pass


class RightPanel(RoundedContainer):
    """Right panel for additional monitoring information."""
    
    def __init__(self, agent: Agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self.selected_index = 0
        self._messages_refresh_timer = None
        self._rendered_messages = []
        self._last_selected_index = 0

    def _get_num_agents(self) -> int:
        try:
            agents = getattr(self.agent, "code_agents", [])
            count = len(agents)
            print(f"🔍 _get_num_agents: found {count} agents")
            return count
        except Exception as e:
            print(f"🔍 _get_num_agents: exception {e}")
            try:
                return int(getattr(self.agent, "cfg").num_code_agents)
            except Exception:
                return 1

    def _get_selected_agent(self):
        agents = getattr(self.agent, "code_agents", None)
        print(f"🔍 _get_selected_agent: agents={type(agents)}, count={len(agents) if agents else 0}, selected_index={self.selected_index}")
        
        if agents and self.selected_index < len(agents):
            agent = agents[self.selected_index]
            print(f"🔍 _get_selected_agent: returning agent {agent.get('name', 'Unknown')}")
            return agent
        
        # Default fallback
        print(f"🔍 _get_selected_agent: using fallback")
        return {
            "name": f"Agent {self.selected_index+1}",
            "messages": [("agent", "Hello"), ("llm", "Hi, here is some code...")],
            "code": "print('hello world')",
            "output_lines": ["hello world"],
        }

    def _refresh_right_panel(self) -> None:
        # Messages (chat bubbles) with minimal updates to avoid flicker
        try:
            messages_list = self.query_one("#messages-list", ListView)
            selected_agent = self._get_selected_agent()
            messages = selected_agent.get("messages", [])
            
            print(f"🔍 Messages debug: selected_agent keys={list(selected_agent.keys())}")
            print(f"🔍 Messages debug: raw messages={messages}, type={type(messages)}, len={len(messages)}")

            normalized: list[tuple[str, str]] = []
            for i, msg in enumerate(messages):
                print(f"🔍 Message {i}: {msg}, type={type(msg)}")
                role, content = (msg[0], msg[1]) if isinstance(msg, tuple) and len(msg) >= 2 else ("llm", str(msg))
                actual_role = "agent" if role == "agent" else "llm"
                normalized.append((actual_role, content))

            print(f"🔍 Normalized messages: {normalized}")

            agent_switched = (self._last_selected_index != self.selected_index)

            if agent_switched or not self._rendered_messages:
                # Full rebuild on first render or when switching agent
                messages_list.remove_children()
                if normalized:
                    items = [ListItem(MessageRow(role, content)) for role, content in normalized]
                    messages_list.mount(*items)
                    print(f"🔄 Mounted {len(items)} message items")
                else:
                    messages_list.mount(ListItem(MessageRow("llm", "No messages yet...")))
                    print("🔄 Mounted 'No messages yet' placeholder")
            else:
                # Append only new messages if the beginning matches
                if len(normalized) > len(self._rendered_messages) and normalized[: len(self._rendered_messages)] == self._rendered_messages:
                    tail = normalized[len(self._rendered_messages) :]
                    if tail:
                        new_items = [ListItem(MessageRow(role, content)) for role, content in tail]
                        messages_list.mount(*new_items)
                        print(f"🔄 Appended {len(new_items)} new message items")
                elif normalized != self._rendered_messages:
                    # Fallback to full rebuild if content diverges
                    messages_list.remove_children()
                    items = [ListItem(MessageRow(role, content)) for role, content in normalized]
                    messages_list.mount(*items)
                    print(f"🔄 Rebuilt with {len(items)} message items")

            self._rendered_messages = normalized
            self._last_selected_index = self.selected_index
            # Keep the view scrolled to the most recent message
            try:
                if self._rendered_messages:
                    messages_list.index = len(self._rendered_messages) - 1
            except Exception:
                pass
        except NoMatches:
            pass

        # Ensure the messages view scrolls to the end so latest messages are visible
        try:
            messages_scroll = self.query_one("#messages-scroll", VerticalScroll)
            messages_scroll.scroll_end(animate=False)
        except NoMatches:
            pass

        # Code
        try:
            code_viewer = self.query_one("#code-content", CodeViewer)
            code_str = self._get_selected_agent().get("code", "")
            code_viewer.code = code_str
            code_viewer.update(Syntax(code_str, "python", theme="monokai", line_numbers=True))
        except NoMatches:
            pass

        # Output
        try:
            output_viewer = self.query_one("#output-viewer", OutputViewer)
            output_lines = self._get_selected_agent().get("output_lines", [])
            print(f"🔍 Output debug: {len(output_lines)} lines, first few: {output_lines[:3] if output_lines else 'None'}")
            output_viewer.set_lines(output_lines)
        except NoMatches:
            pass

    def compose(self) -> ComposeResult:
        with TabbedContent(id="right-tabs", initial="messages"):
            with TabPane("Messages", id="messages"):
                yield ListView(id="messages-list")
            with TabPane("Code", id="code"):
                with VerticalScroll(id="code-scroll"):
                    yield CodeViewer("", id="code-content")
            with TabPane("Output", id="output"):
                with VerticalScroll(id="output-scroll"):
                    yield OutputViewer(id="output-viewer")

    def on_mount(self) -> None:
        self.selected_index = 0
        self._refresh_right_panel()
        # Periodically refresh to show new incoming messages without user interaction
        # Keep a handle so we can stop it if needed
        try:
            self._messages_refresh_timer = self.set_interval(1.0, self._refresh_right_panel)
        except Exception:
            pass

    def on_unmount(self) -> None:
        # Stop periodic refresh when panel is removed
        try:
            if self._messages_refresh_timer is not None:
                self._messages_refresh_timer.stop()
        except Exception:
            pass

    DEFAULT_CSS = ""


class MonitoringApp(App):
    """Main monitoring application."""
    
    TITLE = "ComInd Agent Monitor"
    
    BINDINGS = [
        Binding("q", "back", "Back", priority=True),
        Binding("ctrl+c", "quit", "Quit", priority=True),
    ]
    
    def __init__(self, agent: Agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self._selector_refresh_timer = None
    
    def compose(self) -> ComposeResult:
        with Horizontal(id="main-horizontal"):
            yield LeftPanel(self.agent, id="left-panel")
            with Vertical(id="right-container"):
                yield RightPanel(self.agent, id="right-panel")
                yield MetricsPanel(self.agent, id="metrics-panel")
        
        # Build initial selector options from discovered agents (use folder name `draft_id` when available)
        agents_list = getattr(self.agent, "code_agents", []) or []
        if agents_list:
            options = [
                (clean_coder_name(str(a.get("draft_id") or a.get("name") or f"Agent {i+1}")), str(i))
                for i, a in enumerate(agents_list)
            ]
        else:
            options = [("Agent 1", "0")]
        yield Select(options=options, value="0", id="agent-select", allow_blank=False)

    def action_back(self) -> None:
        """Navigate back to list view if a detail view is active."""
        try:
            left_panel = self.query_one("#left-panel", LeftPanel)
            if left_panel.current_view == "detail":
                left_panel._return_to_list()
        except NoMatches:
            pass

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()
    
    def on_mount(self) -> None:
        # Start periodic refresh for agent selector
        try:
            self._selector_refresh_timer = self.set_interval(5.0, self._refresh_agent_selector)
        except Exception:
            pass
        
        # 等布局结束后再定位，确保能拿到真实尺寸
        self.call_later(self._reposition_agent_select)
    
    def on_unmount(self) -> None:
        # Stop periodic refresh when app is removed
        try:
            if self._selector_refresh_timer is not None:
                self._selector_refresh_timer.stop()
        except Exception:
            pass
    
    def _refresh_agent_selector(self) -> None:
        """Refresh the agent selector with current number of discovered coders."""
        try:
            right_panel = self.query_one("#right-panel", RightPanel)
            # Build options from actual discovered agents; prefer folder name `draft_id`
            code_agents = getattr(getattr(right_panel, "agent", None), "code_agents", []) or []
            if code_agents:
                new_options = [
                    (clean_coder_name(str(a.get("draft_id") or a.get("name") or f"Agent {i+1}")), str(i))
                    for i, a in enumerate(code_agents)
                ]
            else:
                new_options = [("Agent 1", "0")]

            agent_select = self.query_one("#agent-select", Select)
            current_options = len(agent_select.options)

            # Update if option count differs or there are discovered agents (to set proper labels)
            if current_options != len(new_options) or code_agents:
                agent_select.set_options(new_options)

                # Reset selection if current index is out of range
                if right_panel.selected_index >= len(new_options):
                    right_panel.selected_index = 0
                    agent_select.value = "0"

                # Trigger panel refresh
                right_panel._refresh_right_panel()

                print(f"🔄 Agent selector updated: {len(code_agents)} agents")
                
        except Exception as e:
            print(f"Error refreshing agent selector: {e}")
            pass

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "agent-select":
            try:
                idx = int(event.value)
            except Exception:
                idx = 0
            try:
                right_panel = self.query_one("#right-panel", RightPanel)
                right_panel.selected_index = idx
                right_panel._refresh_right_panel()
            except NoMatches:
                pass

    def on_resize(self, event: events.Resize) -> None:
        self._reposition_agent_select()

    def _reposition_agent_select(self) -> None:
        try:
            sel = self.query_one("#agent-select", Select)

            # 真实渲染宽度优先，其次是样式宽度，最后兜底 30
            sel_w = sel.size.width
            if not sel_w:
                # sel.styles.width 可能是一个 Dimension，需要取 .value
                styled_w = getattr(sel.styles.width, "value", None)
                sel_w = int(styled_w) if styled_w else 30

            margin_right = 2  # 右边距（以字符为单位）
            margin_top   = 1  # 上边距

            x = max(0, self.size.width - sel_w - margin_right)
            y = margin_top

            sel.styles.layer = "overlay"
            sel.styles.position = "absolute"
            sel.styles.offset = (x, y)


            # 若你的主题里 height 可能不同，可同步写入高度
            if not sel.size.height:
                styled_h = getattr(sel.styles.height, "value", None)
                if styled_h:
                    sel.styles.height = int(styled_h)

            sel.refresh()
        except NoMatches:
            pass
    
    # 采用类似图片中的深色主题配色
    CSS = """
    Screen {
        layers: base overlay;
        background: #081727;
    }
    
    TabPane > VerticalScroll,
    TabPane > ListView,
    ReportDetailView,
    DatasetDetailView {
        height: 1fr;   
    }

    #left-panel {
        width: 40%;
        height: 100%;
        layer: base;
    }
    
    #right-container {
        width: 60%;
        height: 100%;
        layer: base;
    }
    
    #right-panel {
        width: 100%;
        height: 70%;
        layer: base;
    }
    
    #metrics-panel {
        width: 100%;
        height: 30%;
        layer: base;
        margin-top: 1;
    }
    
    #main-horizontal {
        height: 100%;
        layer: base;
    }
    
    #agent-select {
        layer: overlay;
        padding: 0 1;
        border: round #3b185a;
        background: #081727 70%;
        width: 30;
        height: 5;
    }
    
    RoundedContainer {
        border: round #3b185a;
        background: #081727;
        color: #d7d7d7;
    }
    
    TabbedContent {
        height: 100%;
        background: #081727;
    }
    
    TabPane {
        background: #081727;
    }
    
    Tabs {
        background: #081727;
        height: 3;
        border-bottom: solid #3b185a;
    }
    
    Tab {
        background: #081727;
        color: #d7d7dC;
        border: none;
        height: 3;
        padding: 0 2;
    }
    
    Tab.-active {
        text-style: bold;
    }
    
    ListView {
        height: 100%;
        scrollbar-gutter: stable;
    }
    
    VerticalScroll {
        scrollbar-gutter: stable;
        scrollbar-size-vertical: 1;
        scrollbar-color: #08bd8f;
        scrollbar-color-active: #08db8f;
        scrollbar-color-hover: #08db8f;
    }

    #right-tabs {
        height: 1fr;
    }
    
    #code-scroll, #output-scroll {
        height: 1fr;
        width: 100%;
    }
    
    /* Messages list style overrides to avoid large item boxes */
    #messages-list {
        background: transparent;
    }
    #messages-list ListItem {
        border: none;
        background: transparent;
        padding: 0;
        margin: 0;
        height: auto;
        width: 100%;
    }
    #messages-list ListItem.--highlight,
    #messages-list ListItem.--hover {
        background: transparent;
    }
    /* Fix excessive spacing between messages */
    #messages-list .message-row {
        min-height: 20%;
    }


    /* Chat bubbles */
    .message-row {
        width: 100%;
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
    }

    .spacer {
        width: 1fr;
    }

    .message-bubble {
        width: auto;
        max-width: 85%;
        min-width: 80%;
        height: auto;
        border: round #3b185a;
        padding: 1 2;
        color: #d7d7d7;
    }

    .message-bubble.role-llm {
        background: #0c2a24;
        border: round #08bd8f;
    }

    .message-bubble.role-agent {
        background: #1a0f2b;
        border: round #3b185a;
    }

    .message-row.role-llm {
        content-align: left middle;
    }

    .message-row.role-agent {
        content-align: right middle;
    }

    #messages-container { 
        width: 100%; 
    }
    
    ListItem {
        height: auto;
        background: #081727;
        color: #d7d7d7;
        width: 100%;
        border: round #3b185a;
        padding: 0 1 1 2
    }

    ListItem.--highlight, ListItem.--hover {
        background: #081727;
    }
    
    ListItem Label {
        color: #d7d7d7;
        background: transparent;
        width: 100%;
        text-align: left;
    }
    
    Markdown {
        background: transparent;
        color: #d7d7d7;
    }
    
    IdeaItem {
        border: round #3b185a;
        color: #d7d7d7;
        width: 100%;
        padding: 1;
    }
    
    DirectoryTree {
        background: #081727;
        color: #d7d7d7;
        scrollbar-gutter: stable;
        scrollbar-size-vertical: 1;
        scrollbar-color: #08bd8f;
        scrollbar-color-active: #08db8f;
        scrollbar-color-hover: #08db8f;
    }
    
    #ideas-scroll {
        scrollbar-gutter: stable;
        background: #081727;
    }
    
    #right-panel-placeholder {
        color: #d7d7d7;
    }
    
    #description-container {
        height: 50%;
        border: round #3b185a;
        background: #081727;
        padding: 0 1;
    }
    
    #code-container {
        height: 50%;
        border: round #3b185a;
        background: #081727;
    }
    
    #dataset-description-container {
        height: 50%;
        border: round #3b185a;
        background: #081727;
        padding: 0 1;
    }
    
    #file-tree-container {
        height: 50%;
        border: round #3b185a;
        background: #081727;
    }
    
    #report-detail-container {
        height: 100%;
        width: 100%;
    }
    
    #dataset-detail-container {
        height: 100%;
        width: 100%;
    }
    
    #metrics-title {
        text-style: bold;
        color: #08bd8f;
        background: transparent;
        padding: 0 1;
        height: 1;
    }
    
    #metrics-scroll {
        height: 1fr;
        scrollbar-gutter: stable;
        scrollbar-size-vertical: 1;
        scrollbar-color: #08bd8f;
        scrollbar-color-active: #08db8f;
        scrollbar-color-hover: #08db8f;
    }
    
    #metrics-content {
        background: transparent;
        color: #d7d7d7;
        padding: 0 1;
    }

    #messages-list .message-row {
        min-height: 20%;
        margin-bottom: 0;  /* 你想要的行距，适度即可 */
        padding: 0 1;
    }

    /* —— 确保 ListItem 在消息列表里没有额外边框/内边距 —— */
    #messages-list > ListItem {
        border: none;
        padding: 0;
        margin: 0;
        background: transparent;
    }

    /* 可选：防止布局撑高 */
    #messages-list .spacer {
        min-width: 0;
    }

    /* 选择框本体的外观 */
    #agent-select {
        layer: overlay;
        position: absolute;
        /* 初始先给个大致位置，代码里会根据窗口实时修正 */
        offset: 2 1;
        width: 30;
        height: 5;
        border: round #3b185a;
        background: #081727 70%;
    }
    """


def run_monitor(agent: Agent) -> None:
    """Run the monitoring panel for the given agent."""
    app = MonitoringApp(agent)
    app.run()


if __name__ == "__main__":
    # For testing purposes
    from comind.config import Config
    from pathlib import Path
    
    # Create a dummy config and agent for testing
    config = Config()
    config.competition_id = "test-competition"
    config.competition_input_dir = Path("./test_data")
    config.competition_task_desc = "Test task description"
    
    # This would normally be created through the proper initialization
    # agent = Agent(config)
    # run_monitor(agent)
    print("Monitor module loaded successfully")