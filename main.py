#!/usr/bin/env python3
"""Main entry point for Multi-Agent AI System."""

import asyncio
import argparse
import sys
from typing import Optional

# Add the multi_agent_system directory to the path
sys.path.insert(0, './multi_agent_system')

from multi_agent_system.orchestrator import MultiAgentOrchestrator
from multi_agent_system.enhanced_orchestrator import EnhancedMultiAgentOrchestrator
from multi_agent_system.data_structures import Task, TaskType, TaskPriority
from multi_agent_system.config import config
from multi_agent_system.logging_config import configure_logging


async def run_system(enhanced: bool = False):
    """Initialize and run the multi-agent system."""
    if enhanced:
        print("🚀 Initializing Enhanced Multi-Agent AI System...")
        orchestrator = EnhancedMultiAgentOrchestrator()
    else:
        print("🚀 Initializing Multi-Agent AI System...")
        orchestrator = MultiAgentOrchestrator()
        
    # Configure logging
    configure_logging()
    
    # Initialize orchestrator
    success = await orchestrator.initialize()
    
    if not success:
        print("❌ Failed to initialize system. Please check your API keys and configuration.")
        return None
    
    print("✅ System initialized successfully!")
    
    # Get system status properly without await in f-string
    status = await orchestrator.get_system_status()
    print(f"📊 Available agents: {list(status['agents_available'])}")
    
    return orchestrator


async def process_single_task(orchestrator, task_description: str, 
                            task_type: TaskType = TaskType.RESEARCH, 
                            priority: TaskPriority = TaskPriority.MEDIUM) -> None:
    """Process a single task and display results."""
    print(f"📋 Processing task: {task_description}")
    
    # Create task object
    task = Task(
        type=task_type,
        priority=priority,
        title="User Task",
        description=task_description,
        requirements=[task_description]
    )
    
    # Process the task
    result = await orchestrator.process_task(task)
    
    # Display results
    print(f"🎯 Task completed with confidence: {result.final_response.confidence_score:.2%}")
    print(f"⏱️  Execution time: {result.execution_time:.2f} seconds")
    
    if result.final_response.success:
        print("📄 Result:")
        print("-" * 50)
        print(result.final_response.content)
        print("-" * 50)
        
        if result.artifacts:
            print(f"📁 Generated artifacts: {len(result.artifacts)}")
            for artifact in result.artifacts:
                print(f"   • {artifact}")
    else:
        print(f"❌ Task failed: {result.final_response.error_message}")
    
    return result


async def interactive_mode(orchestrator):
    """Run system in interactive mode."""
    print("🔍 Interactive Mode - Type 'quit' to exit")
    
    while True:
        try:
            user_input = input("💭 Enter your task: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            if not user_input:
                continue
                
            await process_single_task(orchestrator, user_input)
            
        except KeyboardInterrupt:
            print("\\n\\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent AI System for Research & Coding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Research machine learning algorithms"
  %(prog)s --autonomous "Create a web API for managing todo items"
  %(prog)s --interactive
  %(prog)s --cli "Generate a CLI tool for file encryption"
        """
    )
    parser.add_argument('task', nargs='?', help='Task description to process')
    parser.add_argument('--type', choices=['research', 'coding', 'analysis', 'vision', 'planning', 'execution'], 
                       default='research', help='Task type')
    parser.add_argument('--priority', choices=['low', 'medium', 'high', 'critical'], 
                       default='medium', help='Task priority')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Run in interactive mode')
    parser.add_argument('--enhanced', '-e', action='store_true',
                       help='Use enhanced orchestrator with autonomous coding capabilities')
    parser.add_argument('--autonomous', '-a', action='store_true',
                       help='Generate complete application autonomously (implies --enhanced)')
    parser.add_argument('--cli', '-c', action='store_true',
                       help='Use enhanced CLI interface')
    parser.add_argument('--status', '-s', action='store_true',
                       help='Check system status')
    parser.add_argument('--config-check', action='store_true',
                       help='Check configuration')
    
    args = parser.parse_args()
    
    # Handle CLI mode
    if args.cli:
        # Import and run the enhanced CLI
        from multi_agent_system.cli import AutonomousCodingCLI
        cli = AutonomousCodingCLI()
        if args.task:
            await cli.run_single_task(args.task, "AutonomousApp")
        else:
            await cli.interactive_mode()
        return
    
    # Handle status check
    if args.status:
        orchestrator = EnhancedMultiAgentOrchestrator()
        success = await orchestrator.initialize()
        if success:
            status = await orchestrator.get_system_status()
            print("📊 System Status:")
            print(f"   Agents Available: {len(status['agents_available'])}")
            for agent in status['agents_available']:
                print(f"     • {agent.value}")
            await orchestrator.shutdown()
        else:
            print("❌ Failed to initialize system for status check")
        return
    
    # Handle config check
    if args.config_check:
        print("🔧 Configuration Check:")
        print(f"   USE_OPENROUTER: {config.USE_OPENROUTER}")
        print(f"   USE_TESTING_MODE: {config.USE_TESTING_MODE}")
        print(f"   OPENROUTER_API_KEY SET: {bool(config.OPENROUTER_API_KEY)}")
        if config.OPENROUTER_API_KEY:
            print(f"   Key Length: {len(config.OPENROUTER_API_KEY)} characters")
        return
    
    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("💡 Please set your API keys in the .env file:")
        print("   - OPENROUTER_API_KEY (required)")
        print("   - OPENAI_API_KEY (optional for embeddings)")
        return
    
    # Initialize system
    orchestrator = None
    try:
        # Use enhanced orchestrator if requested or if autonomous mode
        use_enhanced = args.enhanced or args.autonomous
        orchestrator = await run_system(use_enhanced)
        if not orchestrator:
            return
        
        if args.autonomous and args.task:
            # Use autonomous application generation
            from multi_agent_system.autonomous_app_generator import generate_autonomous_application
            print(f"🤖 Generating autonomous application: {args.task}")
            result = await generate_autonomous_application(args.task, "AutonomousApp")
            
            print(f"🎯 Application Generation {'Successful' if result.success else 'Failed'}")
            print(f"📊 Confidence Score: {result.confidence:.2%}")
            
            if result.success:
                print(f"📁 Generated Files ({len(result.generated_files)}):")
                for file_path in result.generated_files[:10]:
                    print(f"   • {file_path}")
                if len(result.generated_files) > 10:
                    print(f"   ... and {len(result.generated_files) - 10} more")
                
                print(f"📦 Dependencies ({len(result.dependencies)}):")
                for dep in result.dependencies:
                    print(f"   • {dep}")
                
                print(f"📈 Quality Metrics:")
                print(f"   Overall: {result.quality_metrics['overall']:.2%}")
                print(f"   Readability: {result.quality_metrics['readability']:.2%}")
                print(f"   Maintainability: {result.quality_metrics['maintainability']:.2%}")
                print(f"   Efficiency: {result.quality_metrics['efficiency']:.2%}")
                print(f"   Security: {result.quality_metrics['security']:.2%}")
                print(f"   Testability: {result.quality_metrics['testability']:.2%}")
            else:
                print(f"❌ Generation failed with issues:")
                for issue in result.issues:
                    print(f"   • {issue}")
        elif args.interactive:
            await interactive_mode(orchestrator)
        elif args.task:
            task_type = TaskType(args.type)
            task_priority = TaskPriority(args.priority)
            await process_single_task(orchestrator, args.task, task_type, task_priority)
        else:
            print("🤔 No task provided. Use --interactive mode or provide a task description.")
            print("💡 Examples:")
            print('   python main.py "Research machine learning algorithms"')
            print('   python main.py --autonomous "Create a web API for managing todo items"')
            print("   python main.py --interactive")
            print("   python main.py --cli")
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean shutdown if orchestrator was initialized
        if orchestrator:
            await orchestrator.shutdown()
            print("👋 System shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())