import joblib
import numpy as np
import sys

def main():
    try:
        # Load the saved model
        model = joblib.load('model.joblib')
        
        print('--- Model Terminal Interface ---')
        print('Enter 3 numbers separated by spaces to get a prediction (or type "exit"):')
        
        while True:
            user_input = input('> ')
            if user_input.lower() == 'exit':
                break
            
            try:
                inputs = [float(i) for i in user_input.split()]
                if len(inputs) != 3:
                    print('Please enter exactly 3 numbers.')
                    continue
                
                prediction = model.predict([inputs])
                print(f'Result: {prediction[0]:.4f}')
            except ValueError:
                print('Invalid input! Please enter numbers only (e.g., 0.5 0.1 0.9).')
            
    except Exception as e:
        print(f'Critical Error: {e}')

if __name__ == "__main__":
    main()