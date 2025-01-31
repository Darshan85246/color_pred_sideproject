import random
from collections import defaultdict, deque
import math
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import json
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='color_predictor.log'
)
logger = logging.getLogger(__name__)


class SimpleNN(nn.Module):
    """Simple neural network for color prediction."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        return self.layer3(x)

class EnhancedColorPredictor:
    def __init__(self, save_file='color_history.json'):
        self.save_file = save_file
        self.valid_colors = ['r', 'g', 'b']
        self.history = []
        self.markov_chain = defaultdict(lambda: defaultdict(float))
        self.recent_patterns = []
        self.pattern_success_rate = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.sequence_frequencies = defaultdict(lambda: defaultdict(int))
        self.prediction_history = []
        self.strategy_weights = {
            'markov': 1.0,
            'recent': 1.0,
            'short': 1.0,
            'overall': 1.0,
            'frequency': 1.0
        }
        
        # Neural Network parameters
        self.sequence_length = 10
        self.learning_rate = 0.01
        self.min_learning_rate = 0.001
        self.learning_rate_decay = 0.995
        self.momentum = 0.9
        self.accuracy_window = 50  # Window size for accuracy calculation
        self.training_metrics = []
        self.prediction_history = []
        self.previous_adjustments = defaultdict(float)
        self.accuracy_window = 50
        self.min_weight = 0.1
        
        # Anomaly detection parameters
        self.anomaly_threshold = 3.0
        self.recent_anomalies = deque(maxlen=100)
        self.max_pattern_length = 5
        self.min_pattern_support = 3
        self.min_pattern_confidence = 0.6
        
        # Initialize models
        input_size = self.sequence_length * len(self.valid_colors)
        self.nn_model = SimpleNN(input_size, 64, len(self.valid_colors))
        self.rf_model = RandomForestClassifier(n_estimators=100)
        self.pattern_memory = deque(maxlen=1000)
        
        self.load_history()
        self._initialize_markov_chain()
        
        # Add these new lines right here
        self.prediction_cache = {}
        self.cache_validity_period = 5  # seconds
        self.batch_size = 32
        self.training_buffer = []
        self.model_save_path = 'model_state.pt'
        self.load_models()
        
    def load_models(self):
        """Load saved model states if available."""
        try:
            if os.path.exists(self.model_save_path):
                self.nn_model.load_state_dict(torch.load(self.model_save_path, weights_only=True))
        except Exception as e:
            logger.error(f"Error loading models: {e}")


    def save_models(self):
        """Save model states."""
        try:
            torch.save(self.nn_model.state_dict(), self.model_save_path)
        except Exception as e:
            logger.error(f"Error saving models: {e}")
        
    def _initialize_markov_chain(self):
        """Initialize Markov chain from history."""
        self.markov_chain = defaultdict(lambda: defaultdict(int))
        
        # Process color transitions
        for i in range(len(self.history) - 1):
            current_color = self.history[i]
            next_color = self.history[i + 1]
            self.markov_chain[current_color][next_color] += 1
        
        # Convert counts to probabilities
        for current_color in self.markov_chain:
            total = sum(self.markov_chain[current_color].values())
            if total > 0:  # Avoid division by zero
                for next_color in self.markov_chain[current_color]:
                    self.markov_chain[current_color][next_color] /= total
                    
    def _adjust_learning_rate(self):
        """Adjust learning rate based on recent performance"""
        if len(self.prediction_history) >= self.accuracy_window:
            recent_predictions = self.prediction_history[-self.accuracy_window:]
            correct_predictions = sum(1 for pred in recent_predictions 
                                if pred.get('actual') == pred.get('prediction'))
            accuracy = correct_predictions / len(recent_predictions)
        
            # Decay learning rate if accuracy is stable or improving
            if accuracy > 0.6:  # Threshold for good performance
                self.learning_rate = max(
                    self.min_learning_rate,
                    self.learning_rate * self.learning_rate_decay
                )
            else:
                # Increase learning rate if performance is poor
                self.learning_rate = min(
                    0.1,  # Maximum learning rate cap
                    self.learning_rate / self.learning_rate_decay
            )
                
    def _encode_sequence(self, sequence):
        """One-hot encode color sequence for neural network."""
        encoding = []
        color_to_idx = {color: idx for idx, color in enumerate(self.valid_colors)}
        
        for color in sequence:
            one_hot = [0] * len(self.valid_colors)  # Fixed spacing
            one_hot[color_to_idx[color]] = 1
            encoding.extend(one_hot)
        
        return torch.tensor(encoding, dtype=torch.float32)


    def _update_nn(self, sequence, target):
        """Update neural network with batch training."""
        self.training_buffer.append((sequence, target))
        
        if len(self.training_buffer) >= self.batch_size:
            self.nn_model.train()
            optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=self.learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # Batch processing
            batch_inputs = []
            batch_targets = []
            
            for seq, targ in self.training_buffer:
                if len(seq) >= self.sequence_length:
                    padded_sequence = seq[-self.sequence_length:]
                    batch_inputs.append(self._encode_sequence(padded_sequence))
                    batch_targets.append(targ)
            
            if batch_inputs:
                batch_input_tensor = torch.stack(batch_inputs)
                batch_target_tensor = torch.tensor(batch_targets, dtype=torch.long)
                
                output = self.nn_model(batch_input_tensor)
                loss = criterion(output, batch_target_tensor)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            self.training_buffer = []
            self.save_models()


    def _get_nn_prediction(self):
        """Get prediction from neural network."""
        if len(self.history) < 1:  # Need at least one color
            return None, 0.0
        
        # Pad sequence if needed
        sequence = self.history[-self.sequence_length:]
        while len(sequence) < self.sequence_length:
            sequence.insert(0, self.valid_colors[0])
        
        input_tensor = self._encode_sequence(sequence)
    
        self.nn_model.eval()
        with torch.no_grad():
            output = self.nn_model(input_tensor)
            probabilities = torch.softmax(output, dim=0)
            prediction_idx = torch.argmax(probabilities).item()
            confidence = probabilities[prediction_idx].item()
        
        return self.valid_colors[prediction_idx], confidence


    def _update_rf(self):
        """Update random forest model."""
        if len(self.history) < self.sequence_length + 1:
            return
            
        X = []
        y = []
        
        for i in range(len(self.history) - self.sequence_length):
            sequence = self.history[i:i + self.sequence_length]
            next_color = self.history[i + self.sequence_length]
            
            # Convert colors to numerical features
            features = [self.valid_colors.index(c) for c in sequence]
            X.append(features)
            y.append(self.valid_colors.index(next_color))
            
        self.rf_model.fit(X, y)

    def _get_rf_prediction(self):
        """Get prediction from random forest."""
        if len(self.history) < self.sequence_length:
            return None, 0.0
            
        sequence = self.history[-self.sequence_length:]
        features = [self.valid_colors.index(c) for c in sequence]
        
        try:
            probabilities = self.rf_model.predict_proba([features])[0]
            prediction_idx = np.argmax(probabilities)
            confidence = probabilities[prediction_idx]
            
            return self.valid_colors[prediction_idx], confidence
        except:
            return None, 0.0

    def _predict_from_complex_patterns(self):
        """Predict next color based on complex pattern analysis."""
        patterns = {}
        max_confidence = 0.0
        best_prediction = None
        
        # Analyze patterns of different lengths
        for length in range(2, min(self.max_pattern_length + 1, len(self.history))):
            current_pattern = tuple(self.history[-length:])
            
            # Count pattern occurrences and subsequent colors
            pattern_count = 0
            next_color_counts = defaultdict(int)
            
            for i in range(len(self.history) - length):
                if tuple(self.history[i:i+length]) == current_pattern:
                    pattern_count += 1
                    if i + length < len(self.history):
                        next_color_counts[self.history[i+length]] += 1
                        
            if pattern_count >= self.min_pattern_support:
                for color, count in next_color_counts.items():
                    confidence = count / pattern_count
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_prediction = color
                        
        return best_prediction, max_confidence
    def _get_markov_prediction(self):
        """Get prediction based on Markov chain probabilities."""
        if not self.history:
            return None, 0.0
        
        current_color = self.history[-1]
        if current_color not in self.markov_chain:
            return None, 0.0
        
        # Get probabilities for next color
        next_color_probs = self.markov_chain[current_color]
        if not next_color_probs:
            return None, 0.0
        
        # Find color with highest probability
        next_color = max(next_color_probs.items(), key=lambda x: x[1])
        return next_color[0], next_color[1]

    def _get_time_based_prediction(self):
        """Get prediction based on time patterns."""
        if len(self.history) < self.sequence_length:
            return None, 0.0
        
        current_hour = datetime.now().hour
        hour_patterns = defaultdict(lambda: defaultdict(int))
        
        for i in range(len(self.history) - 1):
            hour_patterns[self.history[i]][self.history[i+1]] += 1
            
        total_patterns = sum(sum(patterns.values()) 
                           for patterns in hour_patterns.values())
        
        if total_patterns == 0:
            return None, 0.0
            
        color_probabilities = defaultdict(float)
        for prev_color in hour_patterns:
            for next_color, count in hour_patterns[prev_color].items():
                color_probabilities[next_color] += count / total_patterns
                
        if not color_probabilities:
            return None, 0.0
            
        best_color = max(color_probabilities.items(), key=lambda x: x[1])
        return best_color


    def _detect_anomalies(self, prediction):
        """Detect anomalous patterns in color sequence."""
        if len(self.history) < self.sequence_length:
            return False
            
        # Calculate frequency of each color in recent history
        recent_frequencies = defaultdict(int)
        for color in self.history[-self.sequence_length:]:
            recent_frequencies[color] += 1
            
        # Calculate expected frequency
        expected_freq = self.sequence_length / len(self.valid_colors)
        
        # Check if prediction would create an anomalous pattern
        predicted_frequencies = recent_frequencies.copy()
        predicted_frequencies[prediction] += 1
        
        for color, freq in predicted_frequencies.items():
            if abs(freq - expected_freq) > self.anomaly_threshold:
                self.recent_anomalies.append({
                    'pattern': self.history[-self.sequence_length:],
                    'prediction': prediction,
                    'timestamp': datetime.now().timestamp()
                })
                return True
                
        return False

    def _calculate_entropy(self, probabilities):
        """Calculate entropy of probability distribution."""
        return -sum(p * math.log2(p) for p in probabilities.values() if p > 0)

    def add_color(self, color):
        """Add new color and update models."""
        if color not in self.valid_colors:
            raise ValueError(f"Invalid color. Must be one of {self.valid_colors}")
            
        self.history.append(color)
        
        # Update models
        self._update_rf()
        self._update_nn(self.history[-self.sequence_length:], self.valid_colors.index(color))
        
        # Save history periodically
        if len(self.history) % 10 == 0:
            self.save_history()
            
    def predict_next_color(self):
        """Enhanced prediction with caching and error handling."""
        if len(self.history) < self.sequence_length:
            return random.choice(self.valid_colors), 0.333

        cache_key = tuple(self.history[-self.sequence_length:])
        current_time = datetime.now().timestamp()
        
        # Check cache
        if cache_key in self.prediction_cache:
            prediction, timestamp = self.prediction_cache[cache_key]
            if current_time - timestamp < self.cache_validity_period:
                return prediction
        
        try:
            # Get predictions from different models
            predictions = {}
            
            markov_pred = self._get_markov_prediction()
            if markov_pred[0]:
                predictions['markov'] = markov_pred

            pattern_pred = self._predict_from_complex_patterns()
            if pattern_pred[0]:
                predictions['pattern'] = pattern_pred

            nn_pred = self._get_nn_prediction()
            if nn_pred[0]:
                predictions['nn'] = nn_pred

            rf_pred = self._get_rf_prediction()
            if rf_pred[0]:
                predictions['rf'] = rf_pred

            time_pred = self._get_time_based_prediction()
            if time_pred:
                predictions['time'] = time_pred

            if not predictions:
                return random.choice(self.valid_colors), 0.333

            # Calculate weighted probabilities
            final_probs = defaultdict(float)
            total_weight = 0
            
            for strategy, (color, confidence) in predictions.items():
                weight = self.strategy_weights.get(strategy, 1.0)
                final_probs[color] += confidence * weight
                total_weight += weight

            if total_weight > 0:
                for color in final_probs:
                    final_probs[color] /= total_weight

            best_color = max(final_probs.items(), key=lambda x: x[1])
            
            # Cache the prediction
            self.prediction_cache[cache_key] = (best_color, current_time)
            
            return best_color
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return random.choice(self.valid_colors), 0.333

    def save_history(self):
        """Save color history to file."""
        data = {
            'history': self.history,
            'markov_chain': dict(self.markov_chain),
            'strategy_weights': dict(self.strategy_weights)
        }
        
        with open(self.save_file, 'w') as f:
            json.dump(data, f)

    def load_history(self):
        """Load color history from file."""
        if not os.path.exists(self.save_file):
            return
            
        try:
            with open(self.save_file, 'r') as f:
                data = json.load(f)
                self.history = data.get('history', [])
                self.markov_chain.update(data.get('markov_chain', {}))
                self.strategy_weights.update(data.get('strategy_weights', {}))
        except (json.JSONDecodeError, FileNotFoundError):
            print("Error loading history file")

    def get_stats(self):
        """Enhanced statistics with performance metrics."""
        if not self.history:
            return "No colors recorded yet."
            
        stats = []
        total_colors = len(self.history)
        color_counts = defaultdict(int)
        
        for color in self.history:
            color_counts[color] += 1
            
        stats.append(f"Total colors recorded: {total_colors}")
        
        # Add strategy weights
        stats.append("\nCurrent strategy weights:")
        for strategy, weight in self.strategy_weights.items():
            stats.append(f"{strategy}: {weight:.3f}")
        
        # Add prediction accuracy
        if self.prediction_history:
            recent_predictions = self.prediction_history[-self.accuracy_window:]
            correct_predictions = sum(1 for pred in recent_predictions 
                                   if pred.get('actual') == pred['prediction'])
            accuracy = (correct_predictions / len(recent_predictions)) * 100 if recent_predictions else 0
            stats.append(f"\nRecent prediction accuracy: {accuracy:.1f}%")
        
        # Add color distribution
        stats.append("\nColor distribution:")
        for color in self.valid_colors:
            count = color_counts[color]
            percentage = (count / total_colors * 100) if total_colors > 0 else 0
            stats.append(f"{color}: {count} times ({percentage:.1f}%)")
        
        # Add performance metrics
        if hasattr(self, 'training_metrics') and self.training_metrics:
            recent_metrics = self.training_metrics[-100:]
            avg_loss = sum(m['loss'] for m in recent_metrics) / len(recent_metrics)
            stats.append(f"\nAverage Loss (last 100): {avg_loss:.4f}")
        
        # Add model confidence metrics
        if self.prediction_history:
            recent_confidence = [p['confidence'] for p in self.prediction_history[-50:]]
            avg_confidence = sum(recent_confidence) / len(recent_confidence)
            stats.append(f"\nAverage Confidence (last 50): {avg_confidence:.2%}")
        
        return "\n".join(stats)


def main():
    predictor = EnhancedColorPredictor()
    logger.info("Starting Enhanced Color Predictor")
    
    try:
        predictor.load_models()
    except Exception as e:
        logger.error(f"Failed to load models: {e}")

    print("Enhanced Color Predictor")
    print("Commands:")
    print("r/g/b: Enter colors")
    print("x: Remove last color")
    print("stats: Show statistics")
    print("quit: Exit program")

    if predictor.history:
        print(f"Loaded {len(predictor.history)} previous colors")

    while True:
        try:
            user_input = input("Enter command: ").lower().strip()
            
            if user_input == 'quit':
                print("Saving and exiting...")
                predictor.save_history()
                predictor.save_models()
                break
            elif user_input == 'stats':
                print(predictor.get_stats())
                continue
            elif user_input == 'x':
                if predictor.history:
                    removed_color = predictor.history.pop()
                    if predictor.prediction_history:
                        predictor.prediction_history.pop()
                    print(f"Removed last color: {removed_color}")
                    predictor.save_history()
                else:
                    print("No colors to remove")
                continue
            elif user_input in ['r', 'g', 'b']:
                prediction, confidence = predictor.predict_next_color()
                if predictor.prediction_history:
                    predictor.prediction_history[-1]['actual'] = user_input
                predictor.add_color(user_input)
                print(f"Prediction: {prediction}")
                print(f"Confidence: {confidence*100:.1f}%")
            else:
                print("Invalid input. Use r, g, b, x, stats, or quit")
        except Exception as e:
            logger.error(f"Error occurred: {str(e)}")
            print("Please try again")
            continue

if __name__ == "__main__":
    main()

