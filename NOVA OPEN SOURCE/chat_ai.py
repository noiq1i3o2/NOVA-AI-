from llama_cpp import Llama
import sys

def main():
    print('Loading Nova (Llama-3 8B) High-Capacity Model...')
    try:
        llm = Llama(
            model_path='llama_model.gguf',
            n_ctx=4096,  # Increased context for more data handling
            verbose=False
        )

        print('\n--- Nova Terminal Interface (Powered by Llama-3) ---')
        print('Nova is ready. Type your prompt below (or "exit" to quit):')

        while True:
            user_input = input('\nUser: ')
            if user_input.lower() in ['exit', 'quit']:
                break

            # System prompt integrated to establish the 'Nova' persona
            prompt = f'<|start_header_id|>system<|end_header_id|>\n\nYou are Nova, a high-performance AI trained on massive datasets. Provide detailed, expert-level responses.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            
            output = llm(
                prompt,
                max_tokens=1024, # Increased output length
                stop=['<|eot_id|>'],
                echo=False
            )

            response = output['choices'][0]['text']
            print(f'\nNova: {response}')

    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    main()