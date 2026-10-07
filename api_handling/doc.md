# Documentation 
## Goals
- one set of functions to interact with all of the ai in the main function 

## Process
- each ai "type" has a module 
- the module is plugged in based on selection in the main program 
- the main program doesn't have to ack the quirks of the individual AI api interaction; IE you can use simple keywords, like `model.api` instead of "https://www.openai.com/api/" etc 

## How It's Done
### 1. AI Model class object 
