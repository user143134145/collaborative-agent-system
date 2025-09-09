"""Enhanced CLI interface for the Multi-Agent Autonomous Coding System."""

import asyncio
import argparse
import sys
import os
from typing import Optional
import json

# Rich imports for beautiful CLI interface
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.syntax import Syntax

# Add the multi_agent_system directory to the path
sys.path.insert(0, './multi_agent_system')

from multi_agent_system.data_structures import Task, TaskType, TaskPriority
from multi_agent_system.autonomous_app_generator import generate_autonomous_application, AutonomousAppResult
from multi_agent_system.enhanced_orchestrator import EnhancedMultiAgentOrchestrator
from multi_agent_system.config import config


console = Console()


class AutonomousCodingCLI:
    """Enhanced CLI interface for the autonomous coding system."""
    
    def __init__(self):
        self.orchestrator = None
        self.console = console
        
    async def initialize(self) -> bool:
        """Initialize the CLI system."""
        self.console.print(Panel("[bold blue]🚀 Initializing Autonomous Coding System...", expand=False))
        
        try:
            # Validate configuration
            config.validate()
        except ValueError as e:
            self.console.print(Panel(f"[bold red]❌ Configuration error: {e}", expand=False))
            self.console.print(Panel("[bold yellow]💡 Please set your API keys in the .env file:", expand=False))
            self.console.print("   • OPENROUTER_API_KEY")
            self.console.print("   • OPENAI_API_KEY (optional for embeddings)")
            return False
        
        self.console.print(Panel("[bold green]✅ System configuration validated!", expand=False))
        return True
    
    async def run_autonomous_application_generation(self, task_description: str, app_name: str = "AutonomousApp") -> Optional[AutonomousAppResult]:
        """Run autonomous application generation with progress tracking."""
        task = Task(
            type=TaskType.EXECUTION,
            priority=TaskPriority.HIGH,
            title=app_name,
            description=task_description,
            requirements=[task_description]
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            # Create tasks for different phases
            overall_task = progress.add_task("[cyan]Generating application...", total=100)
            
            # Simulate progress updates
            progress.update(overall_task, advance=10, description="[cyan]Analyzing requirements...")
            await asyncio.sleep(0.5)
            
            progress.update(overall_task, advance=15, description="[blue]Planning implementation...")
            await asyncio.sleep(0.5)
            
            progress.update(overall_task, advance=20, description="[green]Generating code...")
            await asyncio.sleep(1.0)
            
            progress.update(overall_task, advance=15, description="[yellow]Creating tests...")
            await asyncio.sleep(0.5)
            
            progress.update(overall_task, advance=10, description="[magenta]Validating quality...")
            await asyncio.sleep(0.5)
            
            progress.update(overall_task, advance=15, description="[orange]Formatting code...")
            await asyncio.sleep(0.3)
            
            progress.update(overall_task, advance=15, description="[purple]Executing application...")
            await asyncio.sleep(0.7)
            
            # Actually generate the application
            try:
                result = await generate_autonomous_application(task_description, app_name)
                progress.update(overall_task, completed=100, description="[bold green]✅ Generation complete!")
                return result
            except Exception as e:
                progress.update(overall_task, description=f"[bold red]❌ Generation failed: {e}")
                return None
    
    def display_result(self, result: AutonomousAppResult) -> None:
        """Display the generation result in a beautiful format."""
        if result.success:
            # Success panel
            self.console.print(Panel("[bold green]🎯 Application Generation Successful!", expand=False))
            
            # Confidence score
            confidence_panel = Panel(f"[bold]📊 Confidence Score: {result.confidence:.2%}", expand=False)
            self.console.print(confidence_panel)
            
            # Files generated
            if result.generated_files:
                table = Table(title="📁 Generated Files", show_header=True, header_style="bold magenta")
                table.add_column("File Path", style="cyan")
                table.add_column("Status", style="green")
                
                for file_path in result.generated_files[:10]:
                    table.add_row(file_path, "✅ Generated")
                    
                if len(result.generated_files) > 10:
                    table.add_row(f"... and {len(result.generated_files) - 10} more files", "")
                    
                self.console.print(table)
            
            # Dependencies
            if result.dependencies:
                dep_table = Table(title="📦 Dependencies", show_header=True, header_style="bold yellow")
                dep_table.add_column("Dependency", style="blue")
                dep_table.add_column("Status", style="green")
                
                for dep in result.dependencies:
                    dep_table.add_row(dep, "✅ Resolved")
                    
                self.console.print(dep_table)
            
            # Quality metrics
            if result.quality_metrics:
                metrics_table = Table(title="📈 Quality Metrics", show_header=True, header_style="bold cyan")
                metrics_table.add_column("Metric", style="magenta")
                metrics_table.add_column("Score", style="blue")
                
                for metric, score in result.quality_metrics.items():
                    metrics_table.add_row(metric.capitalize(), f"{score:.2%}")
                    
                self.console.print(metrics_table)
            
            # Execution result
            if result.execution_result and result.execution_result.success:
                execution_panel = Panel(f"[bold green]✅ Application Execution Successful!\n⏱️  Execution Time: {result.execution_result.execution_time:.2f}s", expand=False)
                self.console.print(execution_panel)
                
                if result.execution_result.content:
                    self.console.print(Panel("[bold]📝 Output Preview:", expand=False))
                    output_preview = result.execution_result.content[:500] + ("..." if len(result.execution_result.content) > 500 else "")
                    self.console.print(Markdown(f"```\n{output_preview}\n```"))
            
            # Checkpoints
            if result.checkpoints:
                checkpoint_panel = Panel(f"[bold]📍 Checkpoints ({len(result.checkpoints)}):", expand=False)
                self.console.print(checkpoint_panel)
                
                for i, checkpoint in enumerate(result.checkpoints[:5]):
                    status = "✅" if checkpoint.success else "❌"
                    self.console.print(f"   {status} {checkpoint.checkpoint_type.value} ({checkpoint.execution_time:.2f}s)")
                    
                if len(result.checkpoints) > 5:
                    self.console.print(f"   ... and {len(result.checkpoints) - 5} more")
                    
        else:
            # Failure panel
            self.console.print(Panel("[bold red]❌ Application Generation Failed!", expand=False))
            
            # Issues
            if result.issues:
                self.console.print(Panel("[bold]Issues:", expand=False))
                for issue in result.issues:
                    self.console.print(f"   • [red]{issue}[/red]")
            
            # Suggestions
            if result.suggestions:
                self.console.print(Panel("[bold yellow]💡 Suggestions:", expand=False))
                for suggestion in result.suggestions:
                    self.console.print(f"   • [yellow]{suggestion}[/yellow]")
    
    async def interactive_mode(self):
        """Run system in interactive mode."""
        self.console.print(Panel("[bold blue]💬 Interactive Mode", expand=False))
        self.console.print("[dim]Type 'quit', 'exit', or 'q' to exit[/dim]")
        self.console.print("[dim]Type 'help' for examples[/dim]")
        
        # Welcome message with examples
        welcome_text = Text()
        welcome_text.append("💡 Examples:\n", style="bold yellow")
        welcome_text.append("   • Create a web API that returns current weather data\n", style="dim")
        welcome_text.append("   • Build a data analysis tool that processes CSV files\n", style="dim")
        welcome_text.append("   • Generate a CLI tool for file encryption\n", style="dim")
        self.console.print(Panel(welcome_text, expand=False))
        
        orchestrator = None
        try:
            # Initialize orchestrator
            orchestrator = EnhancedMultiAgentOrchestrator()
            success = await orchestrator.initialize()
            
            if not success:
                self.console.print(Panel("[bold red]❌ Failed to initialize orchestrator", expand=False))
                return
            
            self.console.print(Panel("[bold green]✅ Orchestrator initialized successfully!", expand=False))
            
            while True:
                try:
                    # Get user input with rich prompt
                    user_input = Prompt.ask("\n[bold cyan]💭 Describe your application")
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break
                        
                    if user_input.lower() == 'help':
                        help_text = Text()
                        help_text.append("💡 Examples:\n", style="bold yellow")
                        help_text.append("   • Create a web API that returns current weather data\n", style="dim")
                        help_text.append("   • Build a data analysis tool that processes CSV files\n", style="dim")
                        help_text.append("   • Generate a CLI tool for file encryption\n", style="dim")
                        help_text.append("   • Make a REST API for managing todo items\n", style="dim")
                        help_text.append("   • Create a data visualization dashboard\n", style="dim")
                        self.console.print(Panel(help_text, expand=False))
                        continue
                        
                    if not user_input:
                        continue
                    
                    # Generate a name for the application
                    app_name = f"App_{int(asyncio.get_event_loop().time())}"
                    if len(user_input.split()) > 3:
                        # Use first few words as app name
                        app_name = "_".join(user_input.split()[:3])
                    
                    # Process with loading animation
                    self.console.print(f"\n[bold blue]🚀 Processing: {user_input}")
                    result = await self.run_autonomous_application_generation(user_input, app_name)
                    
                    if result:
                        self.display_result(result)
                    else:
                        self.console.print(Panel("[bold red]❌ Generation failed", expand=False))
                    
                except KeyboardInterrupt:
                    self.console.print("\n\n[bold yellow]👋 Goodbye![/bold yellow]")
                    break
                except Exception as e:
                    self.console.print(Panel(f"[bold red]❌ Error: {e}", expand=False))
                    import traceback
                    traceback.print_exc()
        
        finally:
            # Cleanup
            if orchestrator:
                await orchestrator.shutdown()
                self.console.print(Panel("[bold green]🧹 System shutdown completed", expand=False))
    
    async def run_single_task(self, task_description: str, app_name: str = "AutonomousApp"):
        """Run a single task."""
        self.console.print(Panel(f"[bold]📋 Generating application: {app_name}", expand=False))
        self.console.print(Panel(f"[dim]{task_description}", expand=False))
        
        try:
            # Generate the application with loading animation
            result = await self.run_autonomous_application_generation(task_description, app_name)
            
            if result:
                self.display_result(result)
            else:
                self.console.print(Panel("[bold red]❌ Generation failed", expand=False))
                
        except Exception as e:
            self.console.print(Panel(f"[bold red]💥 Unexpected error during application generation: {e}", expand=False))
            import traceback
            traceback.print_exc()
    
    async def system_status(self):
        """Display system status."""
        self.console.print(Panel("[bold blue]📊 System Status", expand=False))
        
        try:
            # Initialize orchestrator to check status
            orchestrator = EnhancedMultiAgentOrchestrator()
            success = await orchestrator.initialize()
            
            if success:
                status = await orchestrator.get_system_status()
                self.console.print(f"[green]✅ System Status: Online[/green]")
                self.console.print(f"[cyan]Agents Available: {len(status['agents_available'])}[/cyan]")
                for agent in status['agents_available']:
                    self.console.print(f"   • {agent.value}")
                
                # Check Docker
                execution_agent = orchestrator.agents.get('execution')
                if execution_agent:
                    docker_status = await execution_agent.health_check()
                    self.console.print(f"[green]✅ Docker Status: {'Online' if docker_status else 'Offline'}[/green]")
                else:
                    self.console.print("[yellow]⚠️  Docker Status: Not initialized[/yellow]")
                
                await orchestrator.shutdown()
            else:
                self.console.print("[red]❌ System Status: Offline[/red]")
                
        except Exception as e:
            self.console.print(f"[red]❌ Error checking system status: {e}[/red]")
    
    async def config_check(self):
        """Check configuration."""
        self.console.print(Panel("[bold blue]🔧 Configuration Check", expand=False))
        
        # Check API keys
        self.console.print("[bold]API Keys:[/bold]")
        if config.OPENROUTER_API_KEY:
            self.console.print("   [green]✅ OpenRouter API Key: SET[/green]")
        else:
            self.console.print("   [red]❌ OpenRouter API Key: NOT SET[/red]")
        
        if config.OPENAI_API_KEY:
            self.console.print("   [green]✅ OpenAI API Key: SET[/green]")
        else:
            self.console.print("   [yellow]⚠️  OpenAI API Key: NOT SET (optional for embeddings)[/yellow]")
        
        # Check modes
        self.console.print(f"\n[bold]Modes:[/bold]")
        self.console.print(f"   {'[green]✅' if config.USE_OPENROUTER else '[yellow]⚠️'}  OpenRouter Mode: {config.USE_OPENROUTER}[/green]")
        self.console.print(f"   {'[green]✅' if config.USE_TESTING_MODE else '[yellow]⚠️'}  Testing Mode: {config.USE_TESTING_MODE}[/green]")
        
        # Check models
        if config.USE_OPENROUTER:
            self.console.print(f"\n[bold]OpenRouter Models:[/bold]")
            self.console.print(f"   Orchestrator: {config.OPENROUTER_QWEN_ORCHESTRATOR_MODEL}")
            self.console.print(f"   Claude: {config.OPENROUTER_CLAUDE_MODEL}")
            self.console.print(f"   Coder: {config.OPENROUTER_QWEN_CODER_MODEL}")
            self.console.print(f"   Vision: {config.OPENROUTER_QWEN_VISION_MODEL}")


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Autonomous Coding Agent - Generate complete applications autonomously",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Create a web API that returns current weather data"
  %(prog)s --interactive
  %(prog)s --name "WeatherAPI" "Create a REST API for weather forecasts"
  %(prog)s --status
  %(prog)s --config
        """
    )
    
    parser.add_argument(
        'description', 
        nargs='?', 
        help='Description of the application to generate'
    )
    
    parser.add_argument(
        '--name', '-n',
        default='AutonomousApplication',
        help='Name for the generated application'
    )
    
    parser.add_argument(
        '--interactive', '-i', 
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--status', '-s',
        action='store_true',
        help='Check system status'
    )
    
    parser.add_argument(
        '--config', '-c',
        action='store_true',
        help='Check configuration'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='Autonomous Coding Agent 1.0'
    )
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = AutonomousCodingCLI()
    if not await cli.initialize():
        return
    
    if args.status:
        await cli.system_status()
    elif args.config:
        await cli.config_check()
    elif args.interactive:
        await cli.interactive_mode()
    elif args.description:
        await cli.run_single_task(args.description, args.name)
    else:
        console.print(Panel("[bold yellow]🤔 No application description provided.", expand=False))
        console.print("[dim]💡 Use --interactive for interactive mode or provide a description.[/dim]")
        console.print("[dim]📝 Example: python cli.py \"Create a web API that returns current weather data\"[/dim]")
        console.print("[dim]   Or: python cli.py --interactive[/dim]")


if __name__ == "__main__":
    asyncio.run(main())