import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TF logs
import tensorflow as tf
import numpy as np

def main():
    try:
        print('Loading Deep Learning Model...')
        model = tf.keras.models.load_model('ai_model.h5')
        
        print('
--- Deep AI Terminal Interface ---')
        print('Enter 10 numbers separated by spaces (or type "exit"):')
        
        while True:
            user_input = input('> ')
            if user_input.lower() == 'exit':
                break
            
            try:
                vals = [float(i) for i in user_input.split()]
                if len(vals) != 10:
                    print(f'Expected 10 numbers, got {len(vals)}.')
                    continue
                
                prediction = model.predict(np.array([vals]), verbose=0)
                print(f'AI Prediction Result: {prediction[0][0]:.6f}')
            except ValueError:
                print('Invalid input! Please enter 10 numbers.')
            
    except Exception as e:
        print(f'Critical Error: {e}')

if __name__ == "__main__":
    main()