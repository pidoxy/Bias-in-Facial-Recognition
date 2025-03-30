import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import os
import random
import face_recognition
import cv2

# Set page configuration
st.set_page_config(
    page_title="Face Recognition Bias Evaluator",
    page_icon="🧑‍🔬",
    layout="wide"
)

# Define utility functions
def load_fairface_dataset(base_dir, labels_file, max_samples=100):
    """
    Load FairFace dataset images and metadata
    
    Args:
        base_dir: Base directory for the dataset (data/fairface)
        labels_file: Path to CSV file with labels (train_labels.csv or val_labels.csv)
        max_samples: Maximum number of samples to load
    
    Returns:
        pandas DataFrame with image paths and metadata
    """
    # Load CSV data
    df = pd.read_csv(labels_file)
    
    # Limit number of samples if needed
    if max_samples and len(df) > max_samples:
        # Ensure we get a diverse sample across demographics
        samples = []
        for race in df['race'].unique():
            for gender in df['gender'].unique():
                subset = df[(df['race'] == race) & (df['gender'] == gender)]
                n = min(len(subset), max_samples // (len(df['race'].unique()) * len(df['gender'].unique())))
                if n > 0:
                    samples.append(subset.sample(n))
        if samples:
            df = pd.concat(samples)
        else:
            df = df.sample(max_samples)
    
    # Add full image path - handle the case where file column already has train/ or val/ prefix
    df['image_path'] = df['file'].apply(lambda x: os.path.join(base_dir, x))
    
    # Verify images exist
    df['exists'] = df['image_path'].apply(lambda x: os.path.exists(x))
    valid_df = df[df['exists']].drop(columns=['exists'])
    
    if len(valid_df) == 0:
        st.error(f"No valid images found. Looking in: {base_dir}")
        if os.path.exists(base_dir):
            st.info(f"Directory exists but no matching images found. Example paths we're trying:")
            for i, row in df.head(3).iterrows():
                st.code(f"Looking for: {row['image_path']}")
                st.code(f"File exists: {os.path.exists(row['image_path'])}")
        raise FileNotFoundError(f"No valid images found in {base_dir}")
    
    return valid_df

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

# Title and description
st.title("Face Recognition Bias Evaluation Tool")
st.markdown("""
This interactive application demonstrates bias analysis in face recognition models. Upload images or use sample data to evaluate
performance across demographic groups.
""")

# Sidebar for controls
st.sidebar.header("Controls")

# Model selection (in a real app, you'd implement multiple models)
model_option = st.sidebar.selectbox(
    "Select Face Recognition Model",
    ["Simple Face Recognition"]
)

# Threshold for face matching
threshold = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05,
    help="Threshold for determining if two faces match"
)

# Initialize model and evaluator
@st.cache_resource
def load_model():
    return SimpleFaceRecognitionModel()

model = load_model()
evaluator = BiasEvaluator(model)

# Main tabs
tabs = st.tabs(["Bias Visualization", "Image Comparison", "Dataset Explorer"])

# Tab 1: Bias Visualization
with tabs[0]:
    st.header("Bias Visualization")
    st.markdown("""
    This section visualizes the performance of face recognition models across different demographic groups.
    The charts show False Acceptance Rates (FAR) and False Rejection Rates (FRR) for various demographics.
    """)
    
    # Get sample data for visualization
    sample_data = evaluator.generate_sample_data()
    df = pd.DataFrame(sample_data)
    
    # Display error rates by demographic group
    st.subheader("Error Rates by Demographic Group")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set width of bars
    barWidth = 0.3
    
    # Set positions of bars on X axis
    r1 = np.arange(len(df))
    r2 = [x + barWidth for x in r1]
    
    # Create bars
    ax.bar(r1, df['FAR'], width=barWidth, label='FAR', color='skyblue')
    ax.bar(r2, df['FRR'], width=barWidth, label='FRR', color='salmon')
    
    # Add labels and title
    ax.set_xlabel('Demographic Group', fontweight='bold')
    ax.set_ylabel('Error Rate', fontweight='bold')
    ax.set_xticks([r + barWidth/2 for r in range(len(df))])
    ax.set_xticklabels(df['Demographic'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display error rates by race and gender
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Error Rates by Race")
        race_df = df.groupby('Race').mean(numeric_only=True).reset_index()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Set width of bars
        barWidth = 0.3
        
        # Set positions of bars on X axis
        r1 = np.arange(len(race_df))
        r2 = [x + barWidth for x in r1]
        
        # Create bars
        ax.bar(r1, race_df['FAR'], width=barWidth, label='FAR', color='skyblue')
        ax.bar(r2, race_df['FRR'], width=barWidth, label='FRR', color='salmon')
        
        # Add labels and title
        ax.set_xlabel('Race', fontweight='bold')
        ax.set_ylabel('Average Error Rate', fontweight='bold')
        ax.set_xticks([r + barWidth/2 for r in range(len(race_df))])
        ax.set_xticklabels(race_df['Race'])
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Error Rates by Gender")
        gender_df = df.groupby('Gender').mean(numeric_only=True).reset_index()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Set width of bars
        barWidth = 0.3
        
        # Set positions of bars on X axis
        r1 = np.arange(len(gender_df))
        r2 = [x + barWidth for x in r1]
        
        # Create bars
        ax.bar(r1, gender_df['FAR'], width=barWidth, label='FAR', color='skyblue')
        ax.bar(r2, gender_df['FRR'], width=barWidth, label='FRR', color='salmon')
        
        # Add labels and title
        ax.set_xlabel('Gender', fontweight='bold')
        ax.set_ylabel('Average Error Rate', fontweight='bold')
        ax.set_xticks([r + barWidth/2 for r in range(len(gender_df))])
        ax.set_xticklabels(gender_df['Gender'])
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # ROC curves visualization
    st.subheader("ROC Curves by Demographic Group")
    
    # Generate sample ROC data (for demonstration)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate sample ROC curves for each demographic group
    for i, row in df.iterrows():
        demographic = row['Demographic']
        # Create a curve that's better for groups with lower error rates
        # This is just for visualization purposes
        base_fpr = np.linspace(0, 1, 100)
        a = 5 * (1 - (row['FAR'] + row['FRR']) / 2)  # Higher a = better curve
        tpr = [1 / (1 + np.exp(-a * (x - 0.5))) for x in base_fpr]
        auc_val = -np.trapz(tpr, base_fpr)
        
        ax.plot(base_fpr, tpr, label=f"{demographic} (AUC = {auc_val:.3f})")
    
    # Add diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set labels and title
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves by Demographic Group')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Calculate fairness metrics
    fairness_metrics = evaluator.calculate_fairness_metrics(sample_data)
    
    # Display fairness metrics
    st.subheader("Fairness Metrics")
    col3, col4, col5, col6 = st.columns(4)
    
    with col3:
        st.metric("Demographic Parity Diff", f"{fairness_metrics['demographic_parity_diff']:.4f}")
        st.caption("Difference in FRR between gender groups (lower is better)")
    
    with col4:
        st.metric("Disparate Impact Ratio", f"{fairness_metrics['disparate_impact_ratio']:.4f}")
        st.caption("Ratio of lowest to highest FRR across races (closer to 1 is better)")
    
    with col5:
        st.metric("Highest Error Group", fairness_metrics['max_frr_group'])
        st.caption("Demographic group with highest FRR")
    
    with col6:
        st.metric("Lowest Error Group", fairness_metrics['min_frr_group'])
        st.caption("Demographic group with lowest FRR")
    
    # Display raw data
    with st.expander("View Raw Data"):
        st.dataframe(df)

# Tab 2: Image Comparison
with tabs[1]:
    st.header("Face Comparison Tool")
    st.markdown("""
    Upload two images to see how the face recognition model compares them.
    This tool helps understand how similarity scores work in face recognition.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload First Image")
        image1 = st.file_uploader("Choose an image...", key="image1", type=["jpg", "jpeg", "png"])
        
        if image1 is not None:
            img1 = Image.open(image1).convert('RGB')
            st.image(img1, caption="Uploaded Image 1", use_column_width=True)
        else:
            st.info("Please upload an image.")
    
    with col2:
        st.subheader("Upload Second Image")
        image2 = st.file_uploader("Choose an image...", key="image2", type=["jpg", "jpeg", "png"])
        
        if image2 is not None:
            img2 = Image.open(image2).convert('RGB')
            st.image(img2, caption="Uploaded Image 2", use_column_width=True)
        else:
            st.info("Please upload an image.")
    
    if image1 is not None and image2 is not None:
        st.subheader("Face Comparison Results")
        
        # Convert images to numpy arrays
        img1_array = np.array(img1)
        img2_array = np.array(img2)
        
        try:
            # Compare faces
            similarity, is_match = model.compare_faces(img1_array, img2_array, threshold)
            
            # Display results
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.metric("Similarity Score", f"{similarity:.4f}")
            
            with col4:
                st.metric("Threshold", f"{threshold:.2f}")
            
            with col5:
                match_result = "✅ MATCH" if is_match else "❌ NO MATCH"
                st.metric("Match Result", match_result)
            
            # Similarity score visualization
            fig, ax = plt.subplots(figsize=(10, 2))
            
            # Create a simple gauge visualization
            cmap = plt.cm.RdYlGn
            norm = plt.Normalize(0, 1)
            colors = cmap(norm(np.linspace(0, 1, 100)))
            
            ax.barh([0], [1], color=colors, height=0.3)
            ax.axvline(x=similarity, color='black', linestyle='-', linewidth=3)
            ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xticks([0, threshold, 1])
            ax.set_xticklabels(['0', f'Threshold: {threshold:.2f}', '1'])
            ax.set_yticks([])
            ax.set_title('Similarity Score')
            
            # Add annotations
            ax.text(similarity, -0.2, f'Score: {similarity:.2f}', 
                    horizontalalignment='center', verticalalignment='center',
                    bbox=dict(facecolor='white', alpha=0.8))
            
            st.pyplot(fig)
            
            # Show how threshold choice impacts different demographic groups
            st.subheader("Impact of Threshold on Different Demographics")
            st.info("""
            This visualization shows how changing the similarity threshold affects error rates
            for different demographic groups. Note that this is simulated data for demonstration.
            """)
            
            demo_thresholds = [0.3, 0.5, 0.7, 0.9]
            demo_races = ["White", "Black", "Asian", "Indian"]
            
            # Simulated FAR and FRR at different thresholds for different races
            far_data = {
                "White": [0.25, 0.15, 0.05, 0.01],
                "Black": [0.40, 0.25, 0.12, 0.04],
                "Asian": [0.35, 0.20, 0.09, 0.02],
                "Indian": [0.30, 0.18, 0.07, 0.01]
            }
            
            frr_data = {
                "White": [0.01, 0.05, 0.15, 0.35],
                "Black": [0.03, 0.10, 0.25, 0.45],
                "Asian": [0.02, 0.08, 0.18, 0.40],
                "Indian": [0.01, 0.06, 0.16, 0.38]
            }
            
            # Create dataframe
            threshold_df = pd.DataFrame({
                'Threshold': np.repeat(demo_thresholds, len(demo_races)),
                'Race': demo_races * len(demo_thresholds),
                'FAR': [far_data[race][i] for i in range(len(demo_thresholds)) for race in demo_races],
                'FRR': [frr_data[race][i] for i in range(len(demo_thresholds)) for race in demo_races]
            })
            
            # Create visualizations of how threshold affects different groups
            col6, col7 = st.columns(2)
            
            with col6:
                fig, ax = plt.subplots(figsize=(8, 6))
                for race in demo_races:
                    race_data = threshold_df[threshold_df['Race'] == race]
                    ax.plot(race_data['Threshold'], race_data['FAR'], marker='o', label=f"{race}")
                
                # Add current threshold line
                ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
                ax.text(threshold, 0.35, f'Current: {threshold:.2f}', 
                        horizontalalignment='center', verticalalignment='center',
                        bbox=dict(facecolor='white', alpha=0.7))
                
                ax.set_xlabel('Threshold')
                ax.set_ylabel('False Acceptance Rate (FAR)')
                ax.set_title('Impact of Threshold on FAR by Race')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col7:
                fig, ax = plt.subplots(figsize=(8, 6))
                for race in demo_races:
                    race_data = threshold_df[threshold_df['Race'] == race]
                    ax.plot(race_data['Threshold'], race_data['FRR'], marker='o', label=f"{race}")
                
                # Add current threshold line
                ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
                ax.text(threshold, 0.35, f'Current: {threshold:.2f}', 
                        horizontalalignment='center', verticalalignment='center',
                        bbox=dict(facecolor='white', alpha=0.7))
                
                ax.set_xlabel('Threshold')
                ax.set_ylabel('False Rejection Rate (FRR)')
                ax.set_title('Impact of Threshold on FRR by Race')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            st.markdown("""
            #### Observation
            
            The above charts show how changing the similarity threshold affects different demographic groups.
            Note that some groups experience steeper increases in error rates as the threshold changes,
            indicating that a single global threshold may not provide equal performance across all groups.
            """)
        
        except Exception as e:
            st.error(f"Error comparing faces: {str(e)}")
            st.warning("Make sure both images contain clearly visible faces.")

# Tab 3: Dataset Explorer
with tabs[2]:
    st.header("Face Dataset Explorer")
    st.markdown("""
    Explore FairFace dataset images and analyze performance across demographic groups.
    """)
    
    # Dataset selection
    dataset_split = st.radio(
        "Select Dataset Split",
        ["Training", "Validation"]
    )
    
    # Set paths based on selection
    base_dir = "data/fairface"  # Base directory
    
    if dataset_split == "Training":
        labels_file = os.path.join(base_dir, "train_labels.csv")
    else:  # Validation
        labels_file = os.path.join(base_dir, "val_labels.csv")
    
    # Check file existence
    if os.path.exists(labels_file):
        try:
            # Show the first few rows of the CSV
            sample_df = pd.read_csv(labels_file, nrows=3)
            with st.expander("View CSV sample"):
                st.write("Sample of CSV data:")
                st.write(sample_df)
                
                # Check the first file path
                if len(sample_df) > 0:
                    first_file = sample_df.iloc[0]['file']
                    full_path = os.path.join(base_dir, first_file)
                    st.write(f"First file: {first_file}")
                    st.write(f"Full path: {full_path}")
                    st.write(f"File exists: {os.path.exists(full_path)}")
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
    
    # Try to load the dataset
    try:
        face_df = load_fairface_dataset(base_dir, labels_file, max_samples=100)
        
        # Display dataset info
        st.info(f"""
        **FairFace {dataset_split}** dataset
        - {len(face_df)} images loaded
        - {len(face_df['race'].unique())} racial groups
        - {len(face_df['gender'].unique())} gender groups
        - {len(face_df['age'].unique())} age groups
        """)
        
        # Demographic filters
        st.subheader("Demographic Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Get unique races from the dataset
            races = sorted(face_df['race'].unique())
            race_filter = st.multiselect(
                "Race",
                races,
                default=races[:3] if len(races) > 3 else races
            )
        
        with col2:
            # Get unique genders from the dataset
            genders = sorted(face_df['gender'].unique())
            gender_filter = st.multiselect(
                "Gender",
                genders,
                default=genders
            )
        
        with col3:
            # Get unique age groups from the dataset
            ages = sorted(face_df['age'].unique())
            age_filter = st.multiselect(
                "Age Group",
                ages,
                default=ages[:3] if len(ages) > 3 else ages
            )
        
        # Apply filters
        filtered_df = face_df.copy()
        if race_filter:
            filtered_df = filtered_df[filtered_df['race'].isin(race_filter)]
        if gender_filter:
            filtered_df = filtered_df[filtered_df['gender'].isin(gender_filter)]
        if age_filter:
            filtered_df = filtered_df[filtered_df['age'].isin(age_filter)]
        
        # Show sample images
        st.subheader(f"Sample Images ({len(filtered_df)} matching filters)")
        
        if len(filtered_df) == 0:
            st.warning("No images match the selected filters. Please adjust your selection.")
        else:
            # Display up to 8 sample images
            sample_rows = min(8, len(filtered_df))
            samples = filtered_df.sample(sample_rows)
            
            # Create a grid
            cols = st.columns(4)
            
            for i, (_, row) in enumerate(samples.iterrows()):
                with cols[i % 4]:
                    # Load and display the image
                    try:
                        img = Image.open(row['image_path'])
                        
                        # Create caption with demographic info
                        caption = f"Race: {row['race']}\nGender: {row['gender']}\nAge: {row['age']}"
                        
                        # Display the image
                        st.image(img, caption=caption, use_column_width=True)
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
        
        # Show demographic distribution
        st.subheader("Demographic Distribution")
        
        # Race distribution
        race_counts = filtered_df['race'].value_counts().reset_index()
        race_counts.columns = ['Race', 'Count']
        
        # Gender distribution
        gender_counts = filtered_df['gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Plot race distribution
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(race_counts['Race'], race_counts['Count'], color='skyblue')
            ax.set_xlabel('Race')
            ax.set_ylabel('Count')
            ax.set_title('Race Distribution')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Plot gender distribution
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.bar(gender_counts['Gender'], gender_counts['Count'], color='salmon')
            ax.set_xlabel('Gender')
            ax.set_ylabel('Count')
            ax.set_title('Gender Distribution')
            plt.tight_layout()
            st.pyplot(fig)
            
        # Performance analysis (simulated)
        st.subheader("Performance Analysis")
        st.write("This is a simulation of how face recognition performance varies across demographic groups.")
        
        # Create simulated performance data
        performance_data = []
        
        # Group by race and gender
        for race in race_filter:
            for gender in gender_filter:
                # Simulate performance metrics with bias
                # (In a real app, you would use actual model evaluation results)
                accuracy = 0.92 if race == "White" else (0.88 if race == "East Asian" else 0.84)
                if gender == "Female":
                    accuracy -= 0.03  # Simulate lower accuracy for females
                
                # Add random variation
                accuracy += np.random.uniform(-0.02, 0.02)
                accuracy = min(max(accuracy, 0.70), 0.99)  # Keep in reasonable range
                
                far = 0.03 if race == "White" else (0.05 if race == "East Asian" else 0.08)
                frr = 0.04 if race == "White" else (0.07 if race == "East Asian" else 0.10)
                
                # Add to data
                performance_data.append({
                    "Race": race,
                    "Gender": gender,
                    "Accuracy": accuracy,
                    "FAR": far,
                    "FRR": frr
                })
        
        # Create DataFrame
        perf_df = pd.DataFrame(performance_data)
        
        # Create heatmap
        if len(perf_df) > 0:
            # Pivot table for heatmap
            pivot_acc = perf_df.pivot(index="Race", columns="Gender", values="Accuracy")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot_acc, annot=True, cmap="RdYlGn", vmin=0.7, vmax=1.0, fmt=".2f", ax=ax)
            ax.set_title("Face Recognition Accuracy by Demographic Group")
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display the performance data table
            st.write("Performance Metrics by Demographic Group")
            st.dataframe(perf_df.style.format({
                "Accuracy": "{:.2%}",
                "FAR": "{:.2%}",
                "FRR": "{:.2%}"
            }).background_gradient(cmap="RdYlGn", subset=["Accuracy"]).background_gradient(
                cmap="RdYlGn_r", subset=["FAR", "FRR"]))
            
            # Calculate fairness metrics
            max_acc = perf_df["Accuracy"].max()
            min_acc = perf_df["Accuracy"].min()
            max_group = perf_df.loc[perf_df["Accuracy"].idxmax(), ["Race", "Gender"]].values
            min_group = perf_df.loc[perf_df["Accuracy"].idxmin(), ["Race", "Gender"]].values
            
            st.subheader("Fairness Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Accuracy Gap", f"{(max_acc - min_acc):.2%}")
                st.caption(f"Difference between highest and lowest performing groups")
            
            with col2:
                disparate_impact = min_acc/max_acc
                st.metric("Disparate Impact Ratio", f"{disparate_impact:.4f}")
                st.caption(f"Ratio of lowest to highest accuracy (closer to 1 is better)")
            
            # Recommendations based on findings
            st.subheader("Recommendations")
            
            if (max_acc - min_acc) > 0.05 or disparate_impact < 0.95:
                st.warning("""
                ⚠️ Significant performance disparities detected across demographic groups.
                
                Consider implementing:
                1. More diverse training data with better representation of underperforming groups
                2. Demographic-specific thresholds or calibration
                3. Fairness constraints during model training
                """)
            else:
                st.success("""
                ✅ Performance is relatively balanced across demographic groups.
                
                Continue monitoring and testing with diverse datasets.
                """)
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        
        # Offer demo option
        if st.button("Use Demo Data Instead"):
            st.info("Loading demo data with placeholder images for demonstration")
            
            # Create demo data
            demo_data = [
                {"race": "White", "gender": "Male", "age": "20-29"},
                {"race": "Black", "gender": "Female", "age": "30-39"},
                {"race": "East Asian", "gender": "Male", "age": "40-49"},
                {"race": "Indian", "gender": "Female", "age": "20-29"},
                {"race": "Middle Eastern", "gender": "Male", "age": "50-59"},
                {"race": "Latino_Hispanic", "gender": "Female", "age": "10-19"},
                {"race": "Southeast Asian", "gender": "Male", "age": "60-69"},
                {"race": "White", "gender": "Female", "age": "30-39"}
            ]
            
            # Display sample images
            st.subheader("Sample Images (Demo Data)")
            cols = st.columns(4)
            
            for i, data in enumerate(demo_data):
                with cols[i % 4]:
                    # Use placeholder image
                    img_url = f"https://picsum.photos/seed/{i+100}/200/200"
                    st.image(img_url, caption=f"Race: {data['race']}\nGender: {data['gender']}\nAge: {data['age']}")
            
            # Show demographic distribution
            st.subheader("Demographic Distribution (Demo Data)")
            
            # Create mock data
            demo_race_data = pd.DataFrame({
                'Race': ["White", "Black", "East Asian", "Indian", "Middle Eastern", "Latino_Hispanic", "Southeast Asian"],
                'Count': [100, 80, 70, 65, 75, 60, 90]
            })
            
            demo_gender_data = pd.DataFrame({
                'Gender': ["Male", "Female"],
                'Count': [300, 240]
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(demo_race_data['Race'], demo_race_data['Count'], color='skyblue')
                ax.set_xlabel('Race')
                ax.set_ylabel('Count')
                ax.set_title('Race Distribution')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.bar(demo_gender_data['Gender'], demo_gender_data['Count'], color='salmon')
                ax.set_xlabel('Gender')
                ax.set_ylabel('Count')
                ax.set_title('Gender Distribution')
                plt.tight_layout()
                st.pyplot(fig)
            
            # Create simulated performance heatmap
            st.subheader("Performance Analysis (Demo Data)")
            
            # Create data frame for the heatmap
            races = ["White", "Black", "East Asian", "Indian", "Middle Eastern"]
            genders = ["Male", "Female"]
            
            # Sample performance data with bias
            perf_data = []
            for race in races:
                for gender in genders:
                    accuracy = 0.92 if race == "White" else (0.88 if race == "East Asian" else 0.84)
                    if gender == "Female":
                        accuracy -= 0.03
                    
                    # Add random variation
                    accuracy += np.random.uniform(-0.02, 0.02)
                    accuracy = min(max(accuracy, 0.70), 0.99)
                    
                    perf_data.append({
                        "Race": race,
                        "Gender": gender,
                        "Accuracy": accuracy
                    })
            
            demo_perf_df = pd.DataFrame(perf_data)
            pivot_acc = demo_perf_df.pivot(index="Race", columns="Gender", values="Accuracy")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot_acc, annot=True, cmap="RdYlGn", vmin=0.7, vmax=1.0, fmt=".2f", ax=ax)
            ax.set_title("Face Recognition Accuracy by Demographic Group (Demo Data)")
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show recommendations
            st.subheader("Recommendations")
            st.warning("""
            ⚠️ Significant performance disparities detected across demographic groups.
            
            Consider implementing:
            1. More diverse training data with better representation of underperforming groups
            2. Demographic-specific thresholds or calibration
            3. Fairness constraints during model training
            """)
        else:
            st.info("""
            Please ensure the FairFace dataset is organized as follows:
            
            data/fairface/
            ├── train/             # Training images
            ├── val/               # Validation images
            ├── train_labels.csv   # Training labels
            └── val_labels.csv     # Validation labels
            
            The CSV files should have columns for 'file', 'race', 'gender', and 'age'.
            The 'file' column should contain paths like 'train/1.jpg' or 'val/1.jpg'.
            """)

# Add footer
st.markdown("---")
st.caption("Face Recognition Bias Evaluation Tool | Created for research and educational purposes")