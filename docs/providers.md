# **Providers**

Providers act as the intelligent core of the application, responsible for interpreting user prompts, managing context, and communicating with underlying Large Language Models (LLMs). They are essential components that handle the translation of abstract tasks into concrete instructions for specific LLM APIs.

The system is designed with a flexible architecture that allows you to easily switch between built-in providers or implement custom ones to suit your specific requirements.

## **Built-in Providers**

The application comes pre-configured with support for leading LLM providers, ensuring you have access to powerful models right out of the box.

### **OpenAI Provider (Default)**

The OpenAIProvider is the default engine, leveraging OpenAI's robust models like gpt-4o. It is optimized for general-purpose tasks, instruction following, and complex reasoning.

#### **Configuration**

To configure the OpenAI provider, you must set your API key in your environment variables:

OPENAI\_API\_KEY=your\_api\_key\_here

**Default Model:** gpt-4o

### **Anthropic Provider**

The AnthropicProvider integrates with Anthropic's Claude models, such as claude-3-5-sonnet-20240620. Claude models are often preferred for tasks requiring extensive context windows, nuanced understanding, or specific stylistic outputs.

#### **Configuration**

To use the Anthropic provider, set your API key:

ANTHROPIC\_API\_KEY=your\_api\_key\_here

**Default Model:** claude-3-5-sonnet-20240620

## **Creating a Custom Provider**

The architecture is highly extensible. If you need to integrate a specialized LLM, a local model, or a custom API endpoint, you can easily create a custom provider.

### **The LLMProvider Interface**

All providers must implement the abstract LLMProvider class. This ensures consistency and interchangeability across the system.

The core method you must implement is:

async def generate\_response(self, prompt: str, system\_prompt: str) \-\> str:  
    """  
    Generates a response from the LLM.

    Args:  
        prompt (str): The primary input or question from the user.  
        system\_prompt (str): Contextual instructions guiding the model's behavior.

    Returns:  
        str: The generated response from the model.  
    """  
    pass

### **Example: Custom HTTP Provider**

Below is a practical example of implementing a custom provider that communicates with an arbitrary REST API endpoint.

import aiohttp  
from core.providers import LLMProvider

class MyCustomProvider(LLMProvider):  
    """  
    A custom provider communicating with a private LLM endpoint.  
    """  
    def \_\_init\_\_(self, api\_url: str, api\_key: str):  
        self.api\_url \= api\_url  
        self.api\_key \= api\_key

    async def generate\_response(self, prompt: str, system\_prompt: str) \-\> str:  
        headers \= {  
            "Authorization": f"Bearer {self.api\_key}",  
            "Content-Type": "application/json"  
        }  
          
        payload \= {  
            "messages": \[  
                {"role": "system", "content": system\_prompt},  
                {"role": "user", "content": prompt}  
            \]  
        }

        async with aiohttp.ClientSession() as session:  
            async with session.post(  
                self.api\_url,   
                headers=headers,   
                json=payload  
            ) as response:  
                  
                \# Ensure the request was successful  
                response.raise\_for\_status()  
                  
                data \= await response.json()  
                  
                \# Extract and return the relevant text from the response  
                return data\["choices"\]\[0\]\["message"\]\["content"\]

### **Registering Your Custom Provider**

Once your custom provider is defined, you can integrate it into your workflow by instantiating it and passing it to the relevant components (e.g., an Agent or the core application controller) in place of the built-in providers.
