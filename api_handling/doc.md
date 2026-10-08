# Documentation 
## Goals
- one set of functions to interact with all of the ai in the main function 

## Process
- each ai "type" has a module 
- the module is plugged in based on selection in the main program 
- the main program doesn't have to ack the quirks of the individual AI api interaction; IE you can use simple keywords, like `model.api` instead of "https://www.openai.com/api/" etc 

## AI 
AI was used in parts of this project. It's been appropriately labeled. Don't take any of the things it says as fact. Any licenses or other legal documents are not valid in regards to this project. 

## How It's Done

### Files
1. `ai_model_class.py` - defines AI model class 

### 1. AI Model class object 
In `ai_model_class.py`, the 