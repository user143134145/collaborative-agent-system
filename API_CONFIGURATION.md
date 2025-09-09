# 🔐 API Key Configuration Guide

This guide will help you properly configure API keys for your Autonomous Coding Agent system.

## 📋 Current Configuration Status

Your system is currently configured to:
- ✅ Use OpenRouter (`USE_OPENROUTER=true`)
- ✅ Use Testing Mode (`USE_TESTING_MODE=true`)
- ⚠️ Has placeholder API key in `.env`

## 🔧 How to Configure API Keys

### 1. **Get Your Real OpenRouter API Key**

1. Visit [https://openrouter.ai](https://openrouter.ai)
2. Sign up for a free account (or log in if you already have one)
3. Go to **Settings** → **API Keys**
4. Click **Create New Key**
5. Give it a descriptive name like "Multi-Agent-App"
6. Copy the generated key

### 2. **Update Your .env File**

Edit the `.env` file:
```bash
nano /home/admin1/multi-agent-app/multi_agent_system/.env
```

Replace the placeholder key with your real key:
```bash
# Current (placeholder):
OPENROUTER_API_KEY=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

# Should become (your real key):
OPENROUTER_API_KEY=sk-or-YOUR_REAL_KEY_HERE
```

### 3. **Example of a Properly Configured .env File**

```bash
# API Keys
OPENROUTER_API_KEY=sk-or-abc123def456ghi789jkl012mno345pqr678stu901

# API Endpoints
OPENROUTER_API_BASE=https://openrouter.ai/api/v1

# Model Configuration
OPENROUTER_QWEN_ORCHESTRATOR_MODEL=qwen/qwen2.5-72b-instruct:free
OPENROUTER_CLAUDE_MODEL=anthropic/claude-3.5-sonnet
OPENROUTER_QWEN_CODER_MODEL=qwen/qwen3-coder:free
OPENROUTER_QWEN_VISION_MODEL=qwen/qwen2.5-vl-32b-instruct:free

# OpenAI Configuration (for embeddings)
OPENAI_API_KEY=sk-your-openai-key-here  # Optional, can use placeholder for free tier
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002

# Vector Database
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=knowledge_artifacts

# System Configuration
MAX_RETRIES=3
REQUEST_TIMEOUT=60
RATE_LIMIT_REQUESTS_PER_MINUTE=60
LOG_LEVEL=INFO

# Use OpenRouter (set to true to use OpenRouter instead of individual APIs)
USE_OPENROUTER=true

# Use Testing Mode (set to true to use free tier models)
USE_TESTING_MODE=true

# Memory System
MEMORY_SEARCH_TOP_K=5
MEMORY_SIMILARITY_THRESHOLD=0.7
MAX_CONTEXT_TOKENS=1000000

# Cross Validation
CROSS_VALIDATION_THRESHOLD=0.8
MIN_CONFIDENCE_SCORE=0.6
```

### 4. **Verify Your Configuration**

Test if your configuration is correct:
```bash
cd /home/admin1/multi-agent-app
source venv/bin/activate
PYTHONPATH=/home/admin1/multi-agent-app python3 -c "
import sys
sys.path.insert(0, './multi_agent_system')
from multi_agent_system.config import config
print('Configuration Status:')
print(f'  USE_OPENROUTER: {config.USE_OPENROUTER}')
print(f'  USE_TESTING_MODE: {config.USE_TESTING_MODE}')
print(f'  OPENROUTER_API_KEY SET: {bool(config.OPENROUTER_API_KEY)}')
if config.OPENROUTER_API_KEY:
    print(f'  Key Length: {len(config.OPENROUTER_API_KEY)} characters')
    print(f'  Key Starts With: {config.OPENROUTER_API_KEY[:15]}...')
"
```

## 🎯 Free Tier Models Used in Testing Mode

When `USE_TESTING_MODE=true`, the system automatically uses these cost-effective models:

| Agent Type | Model Used | Cost |
|------------|------------|------|
| Orchestrator | `qwen/qwen2.5-72b-instruct:free` | Free |
| Planner (Claude) | `anthropic/claude-3.5-sonnet` | Paid |
| Coder | `qwen/qwen3-coder:free` | Free |
| Vision | `qwen/qwen2.5-vl-32b-instruct:free` | Free |

## 💰 Cost Optimization Tips

1. **Use Testing Mode**: Keep `USE_TESTING_MODE=true` for development
2. **Monitor Usage**: Check your OpenRouter dashboard for usage
3. **Cache Results**: System stores knowledge for reuse
4. **Batch Requests**: Combine multiple tasks when possible

## 🚨 Common Issues & Solutions

### **Authentication Errors**
- **Cause**: Invalid or expired API key
- **Solution**: Generate a new key from OpenRouter dashboard

### **Rate Limiting**
- **Cause**: Too many requests in a short time
- **Solution**: System has built-in rate limiting, but you can adjust in .env

### **Model Not Found**
- **Cause**: Incorrect model name
- **Solution**: Use the free tier model names listed above

## ✅ Verification Steps

After updating your API key:

1. **Test Configuration**:
   ```bash
   cd /home/admin1/multi-agent-app
   source venv/bin/activate
   PYTHONPATH=/home/admin1/multi-agent-app python3 -c "
   import sys
   sys.path.insert(0, './multi_agent_system')
   from multi_agent_system.config import config
   print('API Key Valid:', len(config.OPENROUTER_API_KEY) > 20 if config.OPENROUTER_API_KEY else False)
   "
   ```

2. **Test Simple Agent**:
   ```bash
   cd /home/admin1/multi-agent-app
   source venv/bin/activate
   PYTHONPATH=/home/admin1/multi-agent-app python3 test_openrouter_agents.py
   ```

3. **Test Full System**:
   ```bash
   cd /home/admin1/multi-agent-app
   source venv/bin/activate
   PYTHONPATH=/home/admin1/multi-agent-app python3 test_system_init.py
   ```

Once you've updated your API key, your system will be ready for full autonomous operation!