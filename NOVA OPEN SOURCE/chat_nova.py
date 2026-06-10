from llama_cpp import Llama
import sys

def main():
    print('Loading Nova High-Capacity Model...')
    try:
        nova_engine = Llama(
            model_path='nova_model.gguf',
            n_ctx=4096,
            verbose=False
        )

        print('\n--- Nova Terminal Interface ---')
        print('Nova is ready. Type your prompt below (or "exit" to quit):')

        while True:
            user_input = input('\nUser: ')
            if user_input.lower() in ['exit', 'quit']:
                break

            prompt = f'<|start_header_id|>system<|end_header_id|>\\n\\nYou are Nova, a high-performance AI trained on massive datasets.<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n'
            
            output = nova_engine(prompt, max_tokens=1024, stop=['<|eot_id|>'], echo=False)
            print(f'\nNova: {output["choices"][0]["text"]}')

    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    main()