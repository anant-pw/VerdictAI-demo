# reports/cli_reporter.py
"""
CLI reporter for VerdictAI evaluation results.
Generates summary tables, exports JSON/CSV/Markdown.
"""

from pathlib import Path
from typing import Dict, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from database.models import DatabaseManager
import json
import csv


class CLIReporter:
    """Generate CLI reports from eval results."""
    
    def __init__(self, db_path: str = "verdictai.db"):
        self.db = DatabaseManager(db_path)
        self.console = Console()
    
    def show_run_summary(self, run_id: str):
        """Display summary table for a run."""
        summary = self.db.get_run_summary(run_id)
        
        if not summary:
            self.console.print(f"[red]Run {run_id} not found[/red]")
            return
        
        run_info = summary['run_info']
        
        # Header panel
        title = f"Run: {run_id}"
        info_text = (
            f"Status: {run_info['status']}\n"
            f"Started: {run_info['started_at']}\n"
            f"Tests: {run_info['total_tests']} "
            f"(✅ {run_info['passed_tests']} | ❌ {run_info['failed_tests']})\n"
            f"Pass Rate: {(run_info['passed_tests']/run_info['total_tests']*100):.1f}%"
        )
        
        self.console.print(Panel(info_text, title=title, border_style="green"))
        
        # Metrics table
        if summary['metrics']:
            table = Table(title="Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Avg Score", justify="right", style="magenta")
            table.add_column("Count", justify="right", style="green")
            
            for metric_name, data in summary['metrics'].items():
                table.add_row(
                    metric_name,
                    f"{data['avg']:.2f}",
                    str(data['count'])
                )
            
            self.console.print(table)
    
    def show_test_details(self, run_id: str):
        """Show detailed test case results."""
        tests = self.db.get_test_cases(run_id)
        
        if not tests:
            self.console.print(f"[yellow]No test cases found for run {run_id}[/yellow]")
            return
        
        table = Table(title=f"Test Cases ({len(tests)} total)")
        table.add_column("Test ID", style="cyan")
        table.add_column("Passed", justify="center")
        table.add_column("Output Preview", style="white")
        
        for test in tests:
            passed_icon = "✅" if test['passed'] else "❌"
            output = test.get('actual_output', '')[:60] + "..." if test.get('actual_output') else "N/A"
            
            table.add_row(
                test['test_id'],
                passed_icon,
                output
            )
        
        self.console.print(table)
    
    def export_json(self, run_id: str, output_path: str):
        """Export run results to JSON."""
        summary = self.db.get_run_summary(run_id)
        tests = self.db.get_test_cases(run_id)
        
        data = {
            "run_info": summary.get('run_info', {}),
            "metrics": summary.get('metrics', {}),
            "test_cases": tests
        }
        
        Path(output_path).write_text(json.dumps(data, indent=2))
        self.console.print(f"[green]✓ Exported to {output_path}[/green]")
    
    def export_csv(self, run_id: str, output_path: str):
        """Export test results to CSV."""
        tests = self.db.get_test_cases(run_id)

        if not tests:
            self.console.print(f"[yellow]No test cases to export[/yellow]")
            return

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=tests[0].keys())
            writer.writeheader()
            writer.writerows(tests)

        self.console.print(f"[green]✓ Exported {len(tests)} rows to {output_path}[/green]")
    
    def export_markdown(self, run_id: str, output_path: str):
        """Export summary as Markdown report."""
        summary = self.db.get_run_summary(run_id)
        tests = self.db.get_test_cases(run_id)
        
        if not summary:
            return
        
        run_info = summary['run_info']
        
        md = f"""# VerdictAI Evaluation Report

## Run: {run_id}

**Status:** {run_info['status']}  
**Started:** {run_info['started_at']}  
**Total Tests:** {run_info['total_tests']}  
**Passed:** ✅ {run_info['passed_tests']}  
**Failed:** ❌ {run_info['failed_tests']}  
**Pass Rate:** {(run_info['passed_tests']/run_info['total_tests']*100):.1f}%

## Metrics

"""
        
        if summary['metrics']:
            md += "| Metric | Avg Score | Count |\n"
            md += "|--------|-----------|-------|\n"
            for metric_name, data in summary['metrics'].items():
                md += f"| {metric_name} | {data['avg']:.2f} | {data['count']} |\n"
        
        md += "\n## Test Results\n\n"
        
        for test in tests:
            icon = "✅" if test['passed'] else "❌"
            md += f"### {icon} {test['test_id']}\n\n"
            md += f"**Input:** {test.get('input_data', 'N/A')[:100]}...\n\n"
            if test.get('error_message'):
                md += f"**Error:** {test['error_message']}\n\n"
        
        Path(output_path).write_text(md, encoding='utf-8')
        self.console.print(f"[green]✓ Exported to {output_path}[/green]")
    
    def list_runs(self, limit: int = 10):
        """List recent eval runs."""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT run_id, started_at, status, total_tests, passed_tests, failed_tests
            FROM eval_runs
            ORDER BY started_at DESC
            LIMIT ?
        """, (limit,))
        
        runs = [dict(row) for row in cursor.fetchall()]
        
        if not runs:
            self.console.print("[yellow]No runs found[/yellow]")
            return
        
        table = Table(title=f"Recent Runs (Last {limit})")
        table.add_column("Run ID", style="cyan")
        table.add_column("Started", style="white")
        table.add_column("Status", justify="center")
        table.add_column("Pass Rate", justify="right", style="green")
        
        for run in runs:
            pass_rate = (run['passed_tests'] / run['total_tests'] * 100) if run['total_tests'] > 0 else 0
            status_color = "green" if run['status'] == 'completed' else "yellow"
            
            table.add_row(
                run['run_id'],
                run['started_at'][:19],
                f"[{status_color}]{run['status']}[/{status_color}]",
                f"{pass_rate:.1f}%"
            )
        
        self.console.print(table)


# CLI interface
if __name__ == "__main__":
    import sys
    
    reporter = CLIReporter()
    
    if len(sys.argv) < 2:
        print("Usage: python reports/cli_reporter.py [command] [args]")
        print("\nCommands:")
        print("  list                    - List recent runs")
        print("  summary <run_id>        - Show run summary")
        print("  details <run_id>        - Show test details")
        print("  export-json <run_id> <path>")
        print("  export-csv <run_id> <path>")
        print("  export-md <run_id> <path>")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "list":
        reporter.list_runs()
    elif cmd == "summary" and len(sys.argv) > 2:
        reporter.show_run_summary(sys.argv[2])
    elif cmd == "details" and len(sys.argv) > 2:
        reporter.show_test_details(sys.argv[2])
    elif cmd == "export-json" and len(sys.argv) > 3:
        reporter.export_json(sys.argv[2], sys.argv[3])
    elif cmd == "export-csv" and len(sys.argv) > 3:
        reporter.export_csv(sys.argv[2], sys.argv[3])
    elif cmd == "export-md" and len(sys.argv) > 3:
        reporter.export_markdown(sys.argv[2], sys.argv[3])