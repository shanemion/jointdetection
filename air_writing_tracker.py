import cv2
import mediapipe as mp
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class AirWritingTracker:
    def __init__(self, sentence):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.7,
                                         min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.sentence_data = []  # Array of word arrays
        self.current_word_data = []  # Current word being written
        self.is_writing = False
        self.current_positions = {}
        self.anchor_position = None
        self.global_anchor = None
        self.start_time = None
        self.word_count = 0
        self.last_word_time = 0
        self.word_start_x = None
        self.word_end_x = None
        self.sentence = sentence 
       

    def start(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Air Writing Tracker')

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            key = cv2.waitKey(5) & 0xFF
            if key == ord('w'):
                self.toggle_writing()
            elif key == ord('e'):
                self.end_sentence()
            elif key == 27:  # ESC
                break

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    self.process_landmarks(hand_landmarks)

            self.draw_ui(image)

            cv2.imshow('Air Writing Tracker', image)

        cap.release()
        cv2.destroyAllWindows()

        # Visualize data after the session ends
        if self.sentence_data:
            self.visualize_data(self.sentence_data)
        
        # Clear sentence_data for the next run
        self.sentence_data = []


    def process_landmarks(self, hand_landmarks):
        current_time = time.time()

        # Extract landmark positions
        index_tip = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                              hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                              hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].z])
        
        # Estimate ring position (between MCP and PIP)
        index_mcp = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                              hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                              hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].z])
        index_pip = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
                              hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
                              hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].z])
        ring_position = np.mean([index_mcp, index_pip], axis=0)

        if self.is_writing:
            if self.anchor_position is None:
                self.anchor_position = ring_position
                self.start_time = current_time
                self.word_start_x = ring_position[0]
                if self.global_anchor is None:
                    self.global_anchor = ring_position

            relative_position = ring_position - self.anchor_position
            global_position = ring_position - self.global_anchor
            normalized_position = self.normalize_position(relative_position)

            self.current_word_data.append({
                'time': current_time - self.start_time,
                'position': relative_position.tolist(),
                'global_position': global_position.tolist(),
                'normalized_position': normalized_position.tolist()
            })

            # Update word_end_x
            self.word_end_x = ring_position[0]

            # Check for new word (reset to near anchor position and moved right)
            if np.linalg.norm(relative_position[:2]) < 0.1 and ring_position[0] < self.word_start_x and current_time - self.last_word_time > 1:
                if len(self.current_word_data) > 0:
                    self.sentence_data.append({
                        'word_data': self.current_word_data,
                        'anchor': (self.anchor_position - self.global_anchor).tolist()
                    })
                    self.current_word_data = []
                    self.word_count += 1
                    self.last_word_time = current_time
                self.anchor_position = ring_position  # Update anchor for new word
                self.word_start_x = ring_position[0]

        # Update current positions for UI display
        self.current_positions = {
            'Index Tip': index_tip,
            'Ring Position': ring_position
        }


    def normalize_position(self, position):
        return position / (np.max(np.abs(position)) + 1e-6)  # Added small epsilon to avoid division by zero


    def toggle_writing(self):
        self.is_writing = not self.is_writing
        if self.is_writing:
            self.anchor_position = None
            self.word_count = 0
            self.sentence_data = []
            self.current_word_data = []
            self.last_word_time = 0
        else:
            self.end_sentence()


    def end_sentence(self):
        if self.current_word_data:
            self.sentence_data.append({
                'word_data': self.current_word_data,
                'anchor': (self.anchor_position - self.global_anchor).tolist() if self.anchor_position is not None and self.global_anchor is not None else [0, 0, 0]
            })
        if self.sentence_data:
            # Use the sentence in the filename
            sanitized_sentence = "_".join(self.sentence.split())  # Replace spaces with underscores
            filename = f"{sanitized_sentence}_{time.strftime('%m%d-%H%M%S')}.json"
            self.save_data(filename)
            print(f"Sentence with {len(self.sentence_data)} words saved to {filename}.")
        self.is_writing = False
        self.current_word_data = []
        self.anchor_position = None
   

    def save_data(self, filename):
        data_to_save = {"sentence": self.sentence, "words": []}
        for word in self.sentence_data:
            if isinstance(word, dict):
                data_to_save["words"].append(word)
            elif isinstance(word, list):
                data_to_save["words"].append({
                    'word_data': word,
                    'anchor': [0, 0, 0]  # Default anchor if not available
                })
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        print(f"Data saved to {filename}")


    def draw_ui(self, image):
        height, width, _ = image.shape
        cv2.rectangle(image, (0, 0), (width, 60), (0, 0, 0), -1)
        cv2.putText(image, f"Writing: {'Yes' if self.is_writing else 'No'} | Words: {self.word_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset = 80
        for key, value in self.current_positions.items():
            cv2.putText(image, f"{key}: ({value[0]:.2f}, {value[1]:.2f}, {value[2]:.2f})",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 30


    def visualize_data(self, data):
        fig = plt.figure(figsize=(20, 16))
        
        # 3D plot (global)
        ax1 = fig.add_subplot(221, projection='3d')
        
        # 2D plot (global flattened)
        ax2 = fig.add_subplot(222)
        
        # 3D plot (normalized)
        ax3 = fig.add_subplot(223, projection='3d')
        
        # 2D plot (normalized flattened)
        ax4 = fig.add_subplot(224)
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(data)))
        
        for word_idx, (word_data, color) in enumerate(zip(data, colors)):
            if isinstance(word_data, dict):
                word = word_data['word_data']
                anchor = word_data.get('anchor', [0, 0, 0])
            else:
                word = word_data
                anchor = [0, 0, 0]
            
            # Global positions
            x_global = [point['global_position'][0] for point in word]
            y_global = [point['global_position'][1] for point in word]
            z_global = [point['global_position'][2] for point in word]
            
            # Normalized positions
            x_norm = [point['normalized_position'][0] for point in word]
            y_norm = [point['normalized_position'][1] for point in word]
            z_norm = [point['normalized_position'][2] for point in word]
            
            # Global 3D plot
            ax1.plot(x_global, y_global, z_global, color=color, label=f'Word {word_idx + 1}')
            ax1.scatter(anchor[0], anchor[1], anchor[2], color=color, s=100, marker='o')
            
            # Global 2D plot
            ax2.plot(x_global, y_global, color=color, label=f'Word {word_idx + 1}')
            ax2.scatter(anchor[0], anchor[1], color=color, s=100, marker='o')
            
            # Normalized 3D plot
            ax3.plot(x_norm, y_norm, z_norm, color=color, label=f'Word {word_idx + 1}')
            
            # Normalized 2D plot
            ax4.plot(x_norm, y_norm, color=color, label=f'Word {word_idx + 1}')
        
        # Global 3D plot settings
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Visualization (Global)')
        ax1.legend()
        
        # Global 2D plot settings
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('2D Projection (Global)')
        ax2.legend()
        
        # Normalized 3D plot settings
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_title('3D Visualization (Normalized)')
        ax3.legend()
        
        # Normalized 2D plot settings
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_title('2D Projection (Normalized)')
        ax4.legend()
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()