# 6Agent_App_calc

This project integrates a chain with prompt templates and utilizes the local Ollama Mistral model for tool use, specifically a calculator. 

## Project Structure

```
6Agent_App_calc
├── src
│   ├── main.py
│   ├── agents
│   │   └── calculator_agent.py
│   ├── chains
│   │   └── calculator_chain.py
│   ├── prompts
│   │   └── calculator_prompt.py
│   └── tools
│       └── calculator_tool.py
├── README.md
└── requirements.txt
```

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd 6Agent_App_calc
   ```

2. **Install dependencies**:
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   Execute the main script:
   ```bash
   python src/main.py
   ```

## Usage

- The application will prompt you for a calculation input.
- You can enter expressions like `2 + 2`, `5 * 3`, etc.
- The calculator agent will process your input and return the result.

## Example

1. Start the application.
2. Input: `3 * (4 + 5)`
3. Output: `27`

## Contributing

Feel free to submit issues or pull requests for improvements or bug fixes. 

## License

This project is licensed under the MIT License.