# 🔧 API Key Configuration Guide

## 📋 Required API Keys

To make your autonomous coding system fully functional, you need to configure the following API keys:

### **1. OpenRouter API Key (Required)**
This single key gives you access to all models in your system:
- Qwen2.5-1M Orchestrator
- Claude 3.5 Reasoning
- Qwen3 Coder
- Qwen2.5-VL Vision

### **How to Get OpenRouter API Key:**
1. Visit [https://openrouter.ai](https://openrouter.ai)
2. Sign up for a free account
3. Go to **Settings** → **API Keys**
4. Click **Create New Key**
5. Give it a descriptive name (e.g., "Multi-Agent-System")
6. Copy the generated key

### **2. OpenAI API Key (Optional)**
Used for embeddings in the memory system. You can use the free tier:
- Sign up at [https://platform.openai.com](https://platform.openai.com)
- Get free credits ($5 for new accounts)
- Create an API key in **API Keys** section

## 🛠️ Configuration Steps

### **Step 1: Edit the .env File**
```bash
cd /home/admin1/multi-agent-app/multi_agent_system
nano .env
```

### **Step 2: Replace Placeholder Keys**
Change these placeholder values to your real API keys:

```bash
# OpenRouter API Key (REQUIRED for all AI models)
OPENROUTER_API_KEY=sk-or-YOUR_REAL_OPENROUTER_API_KEY_HERE

# OpenAI API Key (OPTIONAL for embeddings - can use free tier)
OPENAI_API_KEY=sk-YOUR_OPENAI_API_KEY_HERE

# Anthropic API Key (OPTIONAL if using direct Claude access)
ANTHROPIC_API_KEY=sk-ant-YOUR_ANTHROPIC_API_KEY_HERE

# Qwen API Key (OPTIONAL if using direct Qwen access)
QWEN_API_KEY=sk-YOUR_QWEN_API_KEY_HERE
```

### **Example with Real Keys:**
```bash
# OpenRouter API Key (REQUIRED for all AI models)
OPENROUTER_API_KEY=sk-or-abc123def456ghi789jkl012mno345pqr678stu901

# OpenAI API Key (OPTIONAL for embeddings - can use free tier)
OPENAI_API_KEY=sk-proj-xyz789abc123def456ghi789jkl012mno345pqr678

# Anthropic API Key (OPTIONAL if using direct Claude access)
ANTHROPIC_API_KEY=sk-ant-uvw456xyz789abc123def456ghi789jkl012mno3

# Qwen API Key (OPTIONAL if using direct Qwen access)
QWEN_API_KEY=sk-1234567890abcdef1234567890abcdef
```

### **Step 3: Enable Free Tier Mode (Recommended for Testing)**
```bash
# Use free tier models to save costs during development
USE_TESTING_MODE=true

# Use OpenRouter for all model access
USE_OPENROUTER=true
```

### **Step 4: Verify Configuration**
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

## 💰 Cost Optimization

### **Free Tier Models Used in Testing Mode**
When `USE_TESTING_MODE=true`, the system automatically uses:

| Agent Type | Model | Cost |
|------------|-------|------|
| Orchestrator | `qwen/qwen2.5-72b-instruct:free` | Free |
| Coder | `qwen/qwen3-coder:free` | Free |
| Vision | `qwen/qwen2.5-vl-32b-instruct:free` | Free |
| Reasoning | Claude 3.5 Sonnet | Paid (best available) |

### **Estimated Costs**
- **Free Tier Models**: $0 for OpenRouter free tier models
- **Embeddings**: ~$0.0004 per 1000 tokens (very low cost)
- **Paid Models**: $0.000003-$0.000015 per token for premium models

## 🎯 Testing Your Configuration

### **1. Simple Test**
```bash
cd /home/admin1/multi-agent-app
source venv/bin/activate
PYTHONPATH=/home/admin1/multi-agent-app python3 test_system_init.py
```

### **2. Basic Coding Task**
```bash
cd /home/admin1/multi-agent-app
source venv/bin/activate
PYTHONPATH=/home/admin1/multi-agent-app python3 test_basic_coding.py
```

### **3. Interactive Mode**
```bash
cd /home/admin1/multi-agent-app
source venv/bin/activate
PYTHONPATH=/home/admin1/multi-agent-app python3 autonomous_coder.py --interactive
```

### **4. Enhanced Interactive Mode**
```bash
cd /home/admin1/multi-agent-app
source venv/bin/activate
PYTHONPATH=/home/admin1/multi-agent-app python3 autonomous_coder.py --enhanced
```

## 🚨 Common Issues & Solutions

### **Authentication Errors**
- **Cause**: Invalid or expired API key
- **Solution**: Generate a new key and update .env file

### **Rate Limiting**
- **Cause**: Too many requests in a short time
- **Solution**: System has built-in rate limiting, but you can adjust in .env

### **Model Not Found**
- **Cause**: Incorrect model name
- **Solution**: Use the free tier model names listed above

### **Network Issues**
- **Cause**: Internet connectivity problems
- **Solution**: Check network connection and firewall settings

## ✅ Verification Checklist

Before running your system:

- [ ] OpenRouter API key added to .env
- [ ] OpenAI API key added to .env (optional)
- [ ] USE_OPENROUTER=true set in .env
- [ ] USE_TESTING_MODE=true set in .env (recommended for testing)
- [ ] Docker installed and running
- [ ] Virtual environment activated
- [ ] Configuration verified with test script

Once you've completed these steps, your system will be fully functional and ready for autonomous application generation!