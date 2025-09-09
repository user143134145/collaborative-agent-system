#!/usr/bin/env python3
"""Main entry point for the Autonomous Coding Agent system."""

import asyncio
import argparse
import sys
import os
from typing import Optional
import time

# Add the multi_agent_system directory to the path
sys.path.insert(0, './multi_agent_system')

from multi_agent_system.data_structures import Task, TaskType, TaskPriority
from multi_agent_system.autonomous_app_generator import generate_autonomous_application, AutonomousAppResult
from multi_agent_system.enhanced_orchestrator import EnhancedMultiAgentOrchestrator
from multi_agent_system.config import config
from multi_agent_system.logging_config import SystemLogger

# Rich imports for beautiful CLI interface
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.text import Text


console = Console()
logger = SystemLogger("autonomous_coder")


async def run_autonomous_coding_system():
    """Initialize and run the autonomous coding system."""
    console.print(Panel("[bold blue]🚀 Initializing Autonomous Coding Agent System...", expand=False))
    
    # Configure logging
    try:
        # Validate configuration
        config.validate()
    except ValueError as e:
        console.print(Panel(f"[bold red]❌ Configuration error: {e}", expand=False))
        console.print(Panel("[bold yellow]💡 Please set your API keys in the .env file:", expand=False))
        console.print("   • OPENROUTER_API_KEY")
        console.print("   • OPENAI_API_KEY") 
        return False
    
    console.print(Panel("[bold green]✅ System initialized successfully!", expand=False))
    return True


async def generate_app_with_loading_animation(task_description: str, app_name: str = "Autonomous Application") -> Optional[AutonomousAppResult]:
    """Generate an application with loading animation."""
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
        console=console
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


async def display_result(result: AutonomousAppResult) -> None:
    """Display the generation result in a beautiful format."""
    if result.success:
        # Success panel
        console.print(Panel("[bold green]🎯 Application Generation Successful!", expand=False))
        
        # Confidence score
        confidence_panel = Panel(f"[bold]📊 Confidence Score: {result.confidence:.2%}", expand=False)
        console.print(confidence_panel)
        
        # Files generated
        if result.generated_files:
            table = Table(title="📁 Generated Files", show_header=True, header_style="bold magenta")
            table.add_column("File Path", style="cyan")
            table.add_column("Status", style="green")
            
            for file_path in result.generated_files[:10]:
                table.add_row(file_path, "✅ Generated")
                
            if len(result.generated_files) > 10:
                table.add_row(f"... and {len(result.generated_files) - 10} more files", "")
                
            console.print(table)
        
        # Dependencies
        if result.dependencies:
            dep_table = Table(title="📦 Dependencies", show_header=True, header_style="bold yellow")
            dep_table.add_column("Dependency", style="blue")
            dep_table.add_column("Status", style="green")
            
            for dep in result.dependencies:
                dep_table.add_row(dep, "✅ Resolved")
                
            console.print(dep_table)
        
        # Quality metrics
        if result.quality_metrics:
            metrics_table = Table(title="📈 Quality Metrics", show_header=True, header_style="bold cyan")
            metrics_table.add_column("Metric", style="magenta")
            metrics_table.add_column("Score", style="blue")
            
            for metric, score in result.quality_metrics.items():
                metrics_table.add_row(metric.capitalize(), f"{score:.2%}")
                
            console.print(metrics_table)
        
        # Execution result
        if result.execution_result and result.execution_result.success:
            execution_panel = Panel(f"[bold green]✅ Application Execution Successful!\n⏱️  Execution Time: {result.execution_result.execution_time:.2f}s", expand=False)
            console.print(execution_panel)
            
            if result.execution_result.content:
                console.print(Panel("[bold]📝 Output Preview:", expand=False))
                output_preview = result.execution_result.content[:500] + ("..." if len(result.execution_result.content) > 500 else "")
                console.print(Markdown(f"```\n{output_preview}\n```"))
        
        # Checkpoints
        if result.checkpoints:
            checkpoint_panel = Panel(f"[bold]📍 Checkpoints ({len(result.checkpoints)}):", expand=False)
            console.print(checkpoint_panel)
            
            for i, checkpoint in enumerate(result.checkpoints[:5]):
                status = "✅" if checkpoint.success else "❌"
                console.print(f"   {status} {checkpoint.checkpoint_type.value} ({checkpoint.execution_time:.2f}s)")
                
            if len(result.checkpoints) > 5:
                console.print(f"   ... and {len(result.checkpoints) - 5} more")
                
    else:
        # Failure panel
        console.print(Panel("[bold red]❌ Application Generation Failed!", expand=False))
        
        # Issues
        if result.issues:
            console.print(Panel("[bold]Issues:", expand=False))
            for issue in result.issues:
                console.print(f"   • [red]{issue}[/red]")
        
        # Suggestions
        if result.suggestions:
            console.print(Panel("[bold yellow]💡 Suggestions:", expand=False))
            for suggestion in result.suggestions:
                console.print(f"   • [yellow]{suggestion}[/yellow]")


async def enhanced_interactive_mode():
    """Run system in enhanced interactive mode with chatbox interface."""
    console.print(Panel("[bold blue]💬 Enhanced Interactive Mode", expand=False))
    console.print("[dim]Type 'quit', 'exit', or 'q' to exit[/dim]")
    console.print("[dim]Type 'help' for examples[/dim]")
    
    # Welcome message with examples
    welcome_text = Text()
    welcome_text.append("💡 Examples:\n", style="bold yellow")
    welcome_text.append("   • Create a web API that returns current weather data\n", style="dim")
    welcome_text.append("   • Build a data analysis tool that processes CSV files\n", style="dim")
    welcome_text.append("   • Generate a CLI tool for file encryption\n", style="dim")
    console.print(Panel(welcome_text, expand=False))
    
    orchestrator = None
    try:
        # Initialize orchestrator
        orchestrator = EnhancedMultiAgentOrchestrator()
        success = await orchestrator.initialize()
        
        if not success:
            console.print(Panel("[bold red]❌ Failed to initialize orchestrator", expand=False))
            return
        
        console.print(Panel("[bold green]✅ Orchestrator initialized successfully!", expand=False))
        
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
                    console.print(Panel(help_text, expand=False))
                    continue
                    
                if not user_input:
                    continue
                
                # Generate a name for the application
                app_name = f"App_{int(time.time())}"
                if len(user_input.split()) > 3:
                    # Use first few words as app name
                    app_name = "_".join(user_input.split()[:3])
                
                # Process with loading animation
                console.print(f"\n[bold blue]🚀 Processing: {user_input}")
                result = await generate_app_with_loading_animation(user_input, app_name)
                
                if result:
                    await display_result(result)
                else:
                    console.print(Panel("[bold red]❌ Generation failed", expand=False))
                
            except KeyboardInterrupt:
                console.print("\n\n[bold yellow]👋 Goodbye![/bold yellow]")
                break
            except Exception as e:
                console.print(Panel(f"[bold red]❌ Error: {e}", expand=False))
                import traceback
                traceback.print_exc()
    
    finally:
        # Cleanup
        if orchestrator:
            await orchestrator.shutdown()
            console.print(Panel("[bold green]🧹 System shutdown completed", expand=False))


async def interactive_mode():
    """Run system in interactive mode."""
    console.print(Panel("[bold blue]🔍 Interactive Mode - Type 'quit' to exit", expand=False))
    console.print("[dim]💡 Examples:[/dim]")
    console.print("[dim]   - 'Create a web API that returns current weather data'[/dim]")
    console.print("[dim]   - 'Build a data analysis tool that processes CSV files'[/dim]")
    console.print("[dim]   - 'Generate a CLI tool for file encryption'[/dim]")
    
    orchestrator = None
    try:
        # Initialize orchestrator
        orchestrator = EnhancedMultiAgentOrchestrator()
        success = await orchestrator.initialize()
        
        if not success:
            console.print(Panel("[bold red]❌ Failed to initialize orchestrator", expand=False))
            return
        
        console.print(Panel("[bold green]✅ Orchestrator initialized successfully!", expand=False))
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]💭 Describe your application")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if not user_input:
                    continue
                    
                # Generate a name for the application
                app_name = f"App_{int(time.time())}"
                if len(user_input.split()) > 3:
                    # Use first few words as app name
                    app_name = "_".join(user_input.split()[:3])
                
                # Process the task
                console.print(f"[bold blue]📋 Generating application: {app_name}")
                console.print(f"[dim]📝 Requirements: {user_input}")
                
                result = await generate_autonomous_application(user_input, app_name)
                
                if result:
                    await display_result(result)
                else:
                    console.print(Panel("[bold red]❌ Generation failed", expand=False))
                    
            except KeyboardInterrupt:
                console.print("\n\n[bold yellow]👋 Goodbye![/bold yellow]")
                break
            except Exception as e:
                console.print(Panel(f"[bold red]❌ Error: {e}", expand=False))
                import traceback
                traceback.print_exc()
                
    finally:
        # Cleanup
        if orchestrator:
            await orchestrator.shutdown()
            console.print(Panel("[bold green]🧹 System shutdown completed", expand=False))


async def generate_app(task_description: str, app_name: str = "Autonomous Application") -> None:
    """Generate an application autonomously."""
    console.print(Panel(f"[bold]📋 Generating application: {app_name}", expand=False))
    console.print(Panel(f"[dim]{task_description}", expand=False))
    
    try:
        # Generate the application with loading animation
        result = await generate_app_with_loading_animation(task_description, app_name)
        
        if result:
            await display_result(result)
        else:
            console.print(Panel("[bold red]❌ Generation failed", expand=False))
            
    except Exception as e:
        console.print(Panel(f"[bold red]💥 Unexpected error during application generation: {e}", expand=False))
        import traceback
        traceback.print_exc()


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Autonomous Coding Agent - Generate complete applications autonomously",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Create a web API that returns current weather data"
  %(prog)s --interactive
  %(prog)s --enhanced
  %(prog)s --name "WeatherAPI" "Create a REST API for weather forecasts"
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
        '--enhanced', '-e',
        action='store_true',
        help='Run in enhanced interactive mode with chatbox interface'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='Autonomous Coding Agent 1.0'
    )
    
    args = parser.parse_args()
    
    # Initialize system
    if not await run_autonomous_coding_system():
        return
    
    if args.enhanced:
        await enhanced_interactive_mode()
    elif args.interactive:
        await interactive_mode()
    elif args.description:
        await generate_app(args.description, args.name)
    else:
        console.print(Panel("[bold yellow]🤔 No application description provided.", expand=False))
        console.print("[dim]💡 Use --enhanced for chatbox mode, --interactive for simple mode, or provide a description.[/dim]")
        console.print('[dim]📝 Example: python autonomous_coder.py "Create a web API that returns current weather data"[/dim]')
        console.print("[dim]   Or: python autonomous_coder.py --enhanced[/dim]")


if __name__ == "__main__":
    asyncio.run(main())