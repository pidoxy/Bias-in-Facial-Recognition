import numpy as np
import face_recognition
import torch
import torch.nn.functional as F
from PIL import Image

class SimpleFaceRecognitionModel:
    """Simple face recognition model using face_recognition library"""
    
    def get_embedding(self, image_array):
        """Extract face embedding using face_recognition library"""
        face_encodings = face_recognition.face_encodings(image_array)
        if len(face_encodings) > 0:
            return face_encodings[0]
        else:
            # Return zeros if no face detected
            return np.zeros(128)
    
    def compare_faces(self, image1, image2, threshold=0.6):
        """Compare two face images and return similarity score"""
        emb1 = self.get_embedding(image1)
        emb2 = self.get_embedding(image2)
        
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
        
        # Determine if match based on threshold
        is_match = similarity >= threshold
        
        return similarity, is_match

class BiasEvaluator:
    """Class for evaluating bias in face recognition models"""
    
    def __init__(self, model=None):
        """Initialize with a face recognition model"""
        self.model = model if model else SimpleFaceRecognitionModel()
        
    def compute_similarity(self, emb1, emb2):
        """Compute cosine similarity between embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
    
    def generate_sample_data(self):
        """Generate sample performance data for demonstration"""
        # Sample demographic groups
        demographic_groups = ["White_Male", "White_Female", "Black_Male", "Black_Female", 
                             "Asian_Male", "Asian_Female", "Indian_Male", "Indian_Female"]
        
        # Sample FAR values (simulate bias by having lower error rates for certain groups)
        far_values = {
            "White_Male": 0.05,
            "White_Female": 0.08,
            "Black_Male": 0.12,
            "Black_Female": 0.15,
            "Asian_Male": 0.09,
            "Asian_Female": 0.11,
            "Indian_Male": 0.07,
            "Indian_Female": 0.10
        }
        
        # Sample FRR values
        frr_values = {
            "White_Male": 0.03,
            "White_Female": 0.07,
            "Black_Male": 0.10,
            "Black_Female": 0.14,
            "Asian_Male": 0.08,
            "Asian_Female": 0.12,
            "Indian_Male": 0.06,
            "Indian_Female": 0.09
        }
        
        # Create structured data
        data = []
        for group in demographic_groups:
            race, gender = group.split('_')
            data.append({
                'Demographic': group,
                'Race': race,
                'Gender': gender,
                'FAR': far_values[group],
                'FRR': frr_values[group]
            })
        
        return data
    
    def calculate_fairness_metrics(self, performance_data):
        """Calculate fairness metrics from performance data"""
        # Extract demographic groups
        race_groups = {}
        gender_groups = {}
        
        for item in performance_data:
            race = item['Race']
            gender = item['Gender']
            
            # Group by race
            if race not in race_groups:
                race_groups[race] = []
            race_groups[race].append(item)
            
            # Group by gender
            if gender not in gender_groups:
                gender_groups[gender] = []
            gender_groups[gender].append(item)
        
        # Calculate average metrics by race
        race_metrics = {}
        for race, items in race_groups.items():
            race_metrics[race] = {
                'FAR': np.mean([item['FAR'] for item in items]),
                'FRR': np.mean([item['FRR'] for item in items])
            }
        
        # Calculate average metrics by gender
        gender_metrics = {}
        for gender, items in gender_groups.items():
            gender_metrics[gender] = {
                'FAR': np.mean([item['FAR'] for item in items]),
                'FRR': np.mean([item['FRR'] for item in items])
            }
        
        # Calculate disparate impact ratio
        frr_values = [metrics['FRR'] for _, metrics in race_metrics.items()]
        min_frr = min(frr_values)
        max_frr = max(frr_values)
        disparate_impact_ratio = min_frr / max_frr if max_frr > 0 else 1.0
        
        # Calculate demographic parity difference
        if 'Male' in gender_metrics and 'Female' in gender_metrics:
            demographic_parity_diff = abs(gender_metrics['Male']['FRR'] - gender_metrics['Female']['FRR'])
        else:
            demographic_parity_diff = None
        
        # Find groups with highest and lowest FRR
        frrs = [(item['Demographic'], item['FRR']) for item in performance_data]
        max_frr_group = max(frrs, key=lambda x: x[1])[0]
        min_frr_group = min(frrs, key=lambda x: x[1])[0]
        
        return {
            'race_metrics': race_metrics,
            'gender_metrics': gender_metrics,
            'disparate_impact_ratio': disparate_impact_ratio,
            'demographic_parity_diff': demographic_parity_diff,
            'max_frr_group': max_frr_group,
            'min_frr_group': min_frr_group
        }