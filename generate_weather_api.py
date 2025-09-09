#!/usr/bin/env python3
"""Generate a complete Weather API application using the autonomous coding agent."""

import sys
import os
import asyncio

# Add the multi_agent_system directory to the path
sys.path.insert(0, './multi_agent_system')

from multi_agent_system.autonomous_app_generator import generate_autonomous_application


async def create_weather_api_app():
    """Create a complete Weather API application."""
    print("🌤️  Creating Complete Weather API Application")
    print("=" * 50)
    
    # Define the application requirements
    app_description = """
    Create a complete REST API for weather data with the following features:
    
    1. Core Functionality:
       - Get current weather for a city
       - Get weather forecast for next 5 days
       - Get historical weather data
       - Search cities by name
    
    2. Data Sources:
       - Use OpenWeatherMap API as the primary data source
       - Mock data for demonstration without API key
    
    3. API Endpoints:
       - GET /weather/current?city={city_name} - Get current weather
       - GET /weather/forecast?city={city_name} - Get 5-day forecast
       - GET /weather/historical?city={city_name}&days={number} - Get historical data
       - GET /cities/search?name={partial_name} - Search cities
    
    4. Response Format:
       - JSON responses with consistent structure
       - Error handling with appropriate HTTP status codes
       - Rate limiting to prevent abuse
    
    5. Technical Requirements:
       - Python with Flask framework
       - Docker support for containerization
       - Environment variable configuration
       - Comprehensive logging
       - Unit and integration tests
       - API documentation with Swagger/OpenAPI
       - Proper error handling and validation
       - Caching for improved performance
       - Health check endpoints
    """
    
    app_name = "WeatherAPI"
    
    print(f"📝 Application: {app_name}")
    print(f"📋 Description: {app_description[:100]}...")
    print("\n🚀 Generating complete application...")
    
    try:
        # Generate the application
        result = await generate_autonomous_application(app_description, app_name)
        
        print(f"\n✅ Application Generation {'Successful' if result.success else 'Failed'}")
        print(f"📊 Confidence Score: {result.confidence:.2%}")
        print(f"⏱️  Execution Time: {result.execution_time:.2f}s")
        
        if result.success:
            print(f"📁 Generated Files ({len(result.generated_files)}):")
            for file_path in result.generated_files[:10]:  # Show first 10 files
                print(f"   • {file_path}")
            if len(result.generated_files) > 10:
                print(f"   ... and {len(result.generated_files) - 10} more")
            
            print(f"\n📦 Dependencies ({len(result.dependencies)}):")
            for dep in result.dependencies:
                print(f"   • {dep}")
            
            print(f"\n📈 Quality Metrics:")
            print(f"   Overall: {result.quality_metrics['overall']:.2%}")
            print(f"   Readability: {result.quality_metrics['readability']:.2%}")
            print(f"   Maintainability: {result.quality_metrics['maintainability']:.2%}")
            print(f"   Efficiency: {result.quality_metrics['efficiency']:.2%}")
            print(f"   Security: {result.quality_metrics['security']:.2%}")
            print(f"   Testability: {result.quality_metrics['testability']:.2%}")
            
            if result.execution_result and result.execution_result.success:
                print(f"\n✅ Application Execution Successful!")
                print(f"⏱️  Execution Time: {result.execution_result.execution_time:.2f}s")
                if result.execution_result.content:
                    print(f"📝 Output Preview:")
                    output_preview = result.execution_result.content[:500] + ("..." if len(result.execution_result.content) > 500 else "")
                    print(output_preview)
            
            # Show issues and suggestions if any
            if result.issues:
                print(f"\n⚠️  Issues Found:")
                for issue in result.issues:
                    print(f"   • {issue}")
            
            if result.suggestions:
                print(f"\n💡 Suggestions:")
                for suggestion in result.suggestions:
                    print(f"   • {suggestion}")
                    
            # Show generated application structure
            print(f"\n📂 Application Structure:")
            if result.app_structure:
                for path, content in list(result.app_structure.items())[:15]:
                    if isinstance(content, str):
                        print(f"   {path}: {len(content)} characters")
                    else:
                        print(f"   {path}: {type(content).__name__}")
                if len(result.app_structure) > 15:
                    print(f"   ... and {len(result.app_structure) - 15} more files")
            
            # Save the generated application
            await _save_generated_app(result, app_name)
            
        else:
            print(f"❌ Generation failed with issues:")
            for issue in result.issues:
                print(f"   • {issue}")
            
            print(f"\n💡 Suggestions:")
            for suggestion in result.suggestions:
                print(f"   • {suggestion}")
                
    except Exception as e:
        print(f"💥 Unexpected error during application generation: {e}")
        import traceback
        traceback.print_exc()


async def _save_generated_app(result: 'AutonomousAppResult', app_name: str) -> None:
    """Save the generated application to disk."""
    try:
        # Create directory for the application
        app_dir = f"./generated_apps/{app_name.lower().replace(' ', '_')}"
        os.makedirs(app_dir, exist_ok=True)
        
        print(f"\n💾 Saving application to: {app_dir}")
        
        # Save all generated files
        saved_files = 0
        for file_path, content in result.app_structure.items():
            # Create subdirectories if needed
            full_path = os.path.join(app_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Save file content
            with open(full_path, 'w') as f:
                f.write(content if isinstance(content, str) else str(content))
            saved_files += 1
        
        print(f"✅ Saved {saved_files} files successfully!")
        
        # Create a summary file
        summary_path = os.path.join(app_dir, "GENERATION_SUMMARY.md")
        with open(summary_path, 'w') as f:
            f.write(f"# {app_name} - Generation Summary\n\n")
            f.write(f"## Generation Status\n")
            f.write(f"**Success**: {result.success}\n")
            f.write(f"**Confidence**: {result.confidence:.2%}\n")
            f.write(f"**Execution Time**: {result.execution_time:.2f}s\n\n")
            
            f.write(f"## Generated Files\n")
            for file_path in result.generated_files:
                f.write(f"- {file_path}\n")
            f.write("\n")
            
            f.write(f"## Dependencies\n")
            for dep in result.dependencies:
                f.write(f"- {dep}\n")
            f.write("\n")
            
            f.write(f"## Quality Metrics\n")
            for metric, value in result.quality_metrics.items():
                f.write(f"- {metric.capitalize()}: {value:.2%}\n")
            f.write("\n")
            
            if result.issues:
                f.write(f"## Issues\n")
                for issue in result.issues:
                    f.write(f"- {issue}\n")
                f.write("\n")
            
            if result.suggestions:
                f.write(f"## Suggestions\n")
                for suggestion in result.suggestions:
                    f.write(f"- {suggestion}\n")
                f.write("\n")
        
        print(f"📄 Generation summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"⚠️  Failed to save application files: {e}")


if __name__ == "__main__":
    print("🚀 Starting Complete Weather API Application Generation")
    print("-" * 60)
    print("This will generate a complete REST API for weather data")
    print("with current weather, forecasts, historical data, and more!")
    print("-" * 60)
    
    try:
        # Run the application generation
        asyncio.run(create_weather_api_app())
        
        print("\n" + "=" * 60)
        print("🎉 Weather API Application Generation Complete!")
        print("📁 Check the generated_apps/weatherapi/ directory for your application")
        print("📝 Review the GENERATION_SUMMARY.md file for details")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()