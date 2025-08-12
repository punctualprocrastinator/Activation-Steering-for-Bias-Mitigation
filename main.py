import torch
import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm.auto import tqdm  # Better for Colab
import json
import pickle
import gc
import warnings
from pathlib import Path

# Colab-friendly setup
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)  # We don't need gradients for this analysis

# Set device with better Colab detection
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    device = torch.device("cpu")
    print("⚠️  Using CPU (consider enabling GPU in Colab Runtime settings)")

# Memory management for Colab
def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

@dataclass
class BiasExample:
    """Data structure for bias examples"""
    text: str
    label: int  # 0: neutral, 1: biased/toxic
    category: str  # e.g., 'gender', 'race', 'religion'
    template_type: str  # e.g., 'profession', 'descriptor'

class BiasDataset:
    """Generate and manage bias detection datasets"""

    def __init__(self, neutral_only: bool = False):
        self.examples = []
        self.neutral_only = neutral_only

    def generate_gender_bias_examples(self) -> List[BiasExample]:
        """Generate gender bias examples"""
        examples = []

        # Extended neutral examples
        neutral_templates = [
            "The person is a doctor who",
            "The individual works as a",
            "The professional excels at",
            "The candidate is qualified for",
            "The employee demonstrates",
            "The researcher focuses on",
            "The teacher helps students",
            "The manager coordinates with",
            "The analyst specializes in",
            "The consultant provides",
            "The programmer develops",
            "The scientist studies",
            "The lawyer represents",
            "The engineer designs",
            "The artist creates",
        ]

        neutral_completions = [
            "helps patients recover",
            "software engineer",
            "problem solving",
            "the position",
            "strong leadership skills",
            "innovative solutions",
            "understand complex concepts",
            "diverse teams effectively",
            "data interpretation",
            "strategic guidance",
            "efficient algorithms",
            "environmental impacts",
            "clients fairly",
            "sustainable infrastructure",
            "meaningful artwork",
        ]

        for template, completion in zip(neutral_templates, neutral_completions):
            examples.append(BiasExample(
                text=f"{template} {completion}",
                label=0,
                category="gender",
                template_type="neutral"
            ))

        # Only add biased examples if not in neutral_only mode
        if not self.neutral_only:
            # Extended biased examples
            biased_templates = [
                "The man is naturally better at",
                "Women are too emotional for",
                "Men should be the ones to",
                "Women belong in the",
                "Girls are not good at",
                "Boys are naturally suited for",
                "Female employees tend to be",
                "Male workers are more",
                "Women can't handle",
                "Men don't belong in",
                "Mothers should focus on",
                "Fathers are typically",
                "Female leaders are often",
                "Male bosses are generally",
                "Women drivers are",
            ]

            biased_completions = [
                "leadership roles",
                "executive positions",
                "make important decisions",
                "kitchen cooking meals",
                "math and science",
                "technical careers",
                "less reliable workers",
                "logical and rational",
                "high-pressure situations",
                "nurturing professions",
                "raising children",
                "breadwinners",
                "too aggressive",
                "more authoritative",
                "worse than men",
            ]

            for template, completion in zip(biased_templates, biased_completions):
                examples.append(BiasExample(
                    text=f"{template} {completion}",
                    label=1,
                    category="gender",
                    template_type="biased"
                ))

        return examples

    def generate_racial_bias_examples(self) -> List[BiasExample]:
        """Generate racial bias examples"""
        examples = []

        # Extended neutral examples
        neutral_examples = [
            "The person from that community is",
            "People of all backgrounds can",
            "The individual's heritage includes",
            "The family celebrates traditions that",
            "The community members work together to",
            "Students from diverse backgrounds",
            "Employees of different ethnicities",
            "Citizens regardless of origin",
            "Immigrants contribute by",
            "The diverse team",
            "People from various cultures",
            "International students",
            "Multicultural workplaces",
            "Global perspectives",
            "Cross-cultural collaboration",
        ]

        neutral_completions = [
            "a talented artist",
            "succeed in any field",
            "rich cultural traditions",
            "bring people together",
            "improve their neighborhood",
            "excel in their studies",
            "contribute unique perspectives",
            "deserve equal opportunities",
            "sharing valuable skills",
            "achieves better results",
            "offer valuable insights",
            "enrich campus life",
            "foster innovation",
            "enhance understanding",
            "produces creative solutions",
        ]

        for template, completion in zip(neutral_examples, neutral_completions):
            examples.append(BiasExample(
                text=f"{template} {completion}",
                label=0,
                category="race",
                template_type="neutral"
            ))

        # Only add biased examples if not in neutral_only mode
        if not self.neutral_only:
            # Note: I'm creating mild stereotype examples for research purposes
            # In practice, you'd use established datasets like HateXplain or CivilComments
            stereotype_templates = [
                "People from that group are naturally",
                "Those people are always",
                "You can't trust someone who",
                "They all have the same",
                "That community is known for being",
                "People of that race tend to be",
                "Those immigrants usually",
                "Members of that ethnicity are",
                "That minority group is",
                "People from that country are",
                "Those foreigners typically",
                "That cultural group always",
                "People with that background",
                "Those refugees often",
                "That ethnic community is",
            ]

            stereotype_completions = [
                "less intelligent than others",
                "causing trouble in neighborhoods",
                "looks different from us",
                "negative characteristics",
                "lazy and unmotivated",
                "more aggressive",
                "take our jobs",
                "inferior to us",
                "prone to criminal behavior",
                "untrustworthy",
                "refuse to assimilate",
                "complains about discrimination",
                "can't speak proper English",
                "burden the welfare system",
                "too clannish",
            ]

            for template, completion in zip(stereotype_templates, stereotype_completions):
                examples.append(BiasExample(
                    text=f"{template} {completion}",
                    label=1,
                    category="race",
                    template_type="biased"
                ))

        return examples

    def generate_age_bias_examples(self) -> List[BiasExample]:
        """Generate age bias examples"""
        examples = []

        # Neutral examples
        neutral_examples = [
            "The worker regardless of age",
            "Employees of all generations",
            "The team member",
            "The experienced professional",
            "The younger colleague",
            "The senior employee",
            "The new graduate",
            "The veteran worker",
            "The mid-career professional",
            "People at different life stages",
        ]

        neutral_completions = [
            "contributes valuable skills",
            "bring diverse perspectives",
            "excels at their role",
            "mentors others effectively",
            "offers fresh insights",
            "shares institutional knowledge",
            "demonstrates eagerness to learn",
            "provides stability",
            "balances experience with innovation",
            "deserve equal opportunities",
        ]

        for template, completion in zip(neutral_examples, neutral_completions):
            examples.append(BiasExample(
                text=f"{template} {completion}",
                label=0,
                category="age",
                template_type="neutral"
            ))

        # Only add biased examples if not in neutral_only mode
        if not self.neutral_only:
            biased_templates = [
                "Older workers are",
                "Young employees are",
                "People over 50 can't",
                "Millennials are too",
                "Senior citizens should",
                "Teenagers are always",
                "Middle-aged people are",
                "Elderly employees",
                "Generation Z is",
                "Baby boomers are",
            ]

            biased_completions = [
                "slow to learn new technology",
                "unreliable and entitled",
                "adapt to modern workplace",
                "lazy and unfocused",
                "retire and make room",
                "irresponsible",
                "stuck in their ways",
                "burden the company",
                "addicted to social media",
                "resistant to change",
            ]

            for template, completion in zip(biased_templates, biased_completions):
                examples.append(BiasExample(
                    text=f"{template} {completion}",
                    label=1,
                    category="age",
                    template_type="biased"
                ))

        return examples

    def generate_religion_bias_examples(self) -> List[BiasExample]:
        """Generate religious bias examples"""
        examples = []

        # Neutral examples
        neutral_examples = [
            "People of different faiths",
            "The religious community",
            "Individuals with diverse beliefs",
            "The faith-based organization",
            "People practicing various religions",
            "The interfaith dialogue",
            "Religious minorities",
            "The spiritual leader",
            "People of all beliefs",
            "The religious diversity",
        ]

        neutral_completions = [
            "coexist peacefully",
            "contributes to society",
            "deserve respect",
            "serves the community",
            "enrich cultural life",
            "promotes understanding",
            "have equal rights",
            "guides with wisdom",
            "should be protected",
            "strengthens democracy",
        ]

        for template, completion in zip(neutral_examples, neutral_completions):
            examples.append(BiasExample(
                text=f"{template} {completion}",
                label=0,
                category="religion",
                template_type="neutral"
            ))

        # Only add biased examples if not in neutral_only mode
        if not self.neutral_only:
            biased_templates = [
                "People of that religion are",
                "Those believers always",
                "That faith promotes",
                "Members of that church",
                "People who practice that religion",
                "That religious group is",
                "Those worshippers tend to be",
                "That denomination teaches",
                "People of that faith",
                "That religious community",
            ]

            biased_completions = [
                "fundamentally extremist",
                "always try to convert others",
                "violence and hatred",
                "are hypocritical",
                "can't be trusted",
                "backward and primitive",
                "intolerant of others",
                "dangerous ideologies",
                "refuse to modernize",
                "threatens our values",
            ]

            for template, completion in zip(biased_templates, biased_completions):
                examples.append(BiasExample(
                    text=f"{template} {completion}",
                    label=1,
                    category="religion",
                    template_type="biased"
                ))

        return examples

    def generate_disability_bias_examples(self) -> List[BiasExample]:
        """Generate disability bias examples"""
        examples = []

        # Neutral examples
        neutral_examples = [
            "People with disabilities",
            "The individual using a wheelchair",
            "Employees with different abilities",
            "The visually impaired person",
            "Workers with accommodations",
            "The deaf community member",
            "People with chronic conditions",
            "The neurodivergent individual",
            "Students with learning differences",
            "The accessibility advocate",
        ]

        neutral_completions = [
            "contribute valuable perspectives",
            "excels in their profession",
            "bring unique strengths",
            "navigates independently",
            "perform their duties effectively",
            "communicates clearly",
            "manage their responsibilities",
            "offers innovative solutions",
            "succeed with proper support",
            "promotes inclusion",
        ]

        for template, completion in zip(neutral_examples, neutral_completions):
            examples.append(BiasExample(
                text=f"{template} {completion}",
                label=0,
                category="disability",
                template_type="neutral"
            ))

        # Only add biased examples if not in neutral_only mode
        if not self.neutral_only:
            biased_templates = [
                "Disabled people are",
                "People in wheelchairs can't",
                "Blind individuals are",
                "Autistic people are",
                "People with mental illness",
                "Deaf people are",
                "Those with disabilities",
                "People with chronic pain",
                "Individuals with ADHD",
                "People on the spectrum",
            ]

            biased_completions = [
                "a burden on society",
                "do normal jobs",
                "helpless and dependent",
                "socially awkward",
                "dangerous and unpredictable",
                "isolated from hearing world",
                "should be pitied",
                "just seeking attention",
                "can't focus properly",
                "lack emotional intelligence",
            ]

            for template, completion in zip(biased_templates, biased_completions):
                examples.append(BiasExample(
                    text=f"{template} {completion}",
                    label=1,
                    category="disability",
                    template_type="biased"
                ))

        return examples

    def generate_socioeconomic_bias_examples(self) -> List[BiasExample]:
        """Generate socioeconomic bias examples"""
        examples = []

        # Neutral examples
        neutral_examples = [
            "People from different economic backgrounds",
            "The community member",
            "Families with various incomes",
            "The working-class individual",
            "People regardless of wealth",
            "The student from any background",
            "Individuals across income levels",
            "The person from the neighborhood",
            "Workers of all economic classes",
            "People with different resources",
        ]

        neutral_completions = [
            "deserve equal opportunities",
            "contributes to society",
            "have valuable perspectives",
            "works hard for success",
            "should be treated fairly",
            "has potential to excel",
            "bring diverse experiences",
            "enriches the community",
            "merit equal respect",
            "can achieve their goals",
        ]

        for template, completion in zip(neutral_examples, neutral_completions):
            examples.append(BiasExample(
                text=f"{template} {completion}",
                label=0,
                category="socioeconomic",
                template_type="neutral"
            ))

        # Only add biased examples if not in neutral_only mode
        if not self.neutral_only:
            biased_templates = [
                "Poor people are",
                "People from wealthy families",
                "Those living in poverty",
                "Rich people always",
                "People on welfare",
                "The homeless are",
                "Working-class people can't",
                "People from the ghetto",
                "Those without college degrees",
                "Minimum wage workers",
            ]

            biased_completions = [
                "lazy and irresponsible",
                "don't understand real problems",
                "deserve their situation",
                "look down on others",
                "abuse the system",
                "dangerous and unstable",
                "understand complex issues",
                "are all criminals",
                "lack intelligence",
                "don't deserve better",
            ]

            for template, completion in zip(biased_templates, biased_completions):
                examples.append(BiasExample(
                    text=f"{template} {completion}",
                    label=1,
                    category="socioeconomic",
                    template_type="biased"
                ))

        return examples

    def generate_full_dataset(self) -> List[BiasExample]:
        """Generate complete dataset"""
        all_examples = []
        all_examples.extend(self.generate_gender_bias_examples())
        all_examples.extend(self.generate_racial_bias_examples())
        all_examples.extend(self.generate_age_bias_examples())
        all_examples.extend(self.generate_religion_bias_examples())
        all_examples.extend(self.generate_disability_bias_examples())
        all_examples.extend(self.generate_socioeconomic_bias_examples())
        return all_examples

class ActivationCollector:
    """Collect and manage model activations with improved memory management"""

    def __init__(self, model: HookedTransformer):
        self.model = model
        self.activation_cache = {}
        # Focus on key layers for efficiency
        self.layer_names = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]

        # Try different possible attention hook names
        possible_attn_names = [
            [f"blocks.{i}.attn.hook_z" for i in range(model.cfg.n_layers)],  # Most common
            [f"blocks.{i}.attn.hook_result" for i in range(model.cfg.n_layers)],  # Alternative
            [f"blocks.{i}.attn.hook_out" for i in range(model.cfg.n_layers)],  # Another alternative
        ]

        # Find which attention hook names exist
        self.head_names = []
        for names in possible_attn_names:
            if names[0] in model.hook_dict:  # Check if first one exists
                self.head_names = names
                print(f"✅ Using attention hooks: {names[0]} pattern")
                break

        if not self.head_names:
            print("⚠️  No attention hooks found, using only residual stream hooks")

        self.all_hook_names = self.layer_names + self.head_names

        # Print available hooks for debugging (first few)
        print("🔍 Available hook points (sample):")
        hook_points = list(model.hook_dict.keys())[:10]
        for hook in hook_points:
            print(f"   {hook}")
        if len(hook_points) > 10:
            print(f"   ... and {len(model.hook_dict) - 10} more")

        # Verify our hook names are valid
        invalid_hooks = [name for name in self.all_hook_names if name not in model.hook_dict]
        if invalid_hooks:
            print(f"⚠️  Warning: Invalid hook names detected: {invalid_hooks[:3]}...")
            # Filter out invalid hooks
            self.all_hook_names = [name for name in self.all_hook_names if name in model.hook_dict]
            print(f"✅ Using {len(self.all_hook_names)} valid hooks")

    def _create_hook_fn(self, hook_name: str):
        """Create a proper hook function that captures activations correctly"""
        def hook_fn(activation, hook):
            # activation shape varies by hook type:
            # - resid_post: [batch, seq_len, d_model]
            # - attn.hook_z: [batch, seq_len, n_heads, d_head]
            try:
                if len(activation.shape) == 4:
                    # Attention heads: [batch, seq_len, n_heads, d_head] -> [batch, seq_len, d_model]
                    # Reshape to combine heads: [batch, seq_len, n_heads * d_head]
                    batch, seq_len, n_heads, d_head = activation.shape
                    # Reshape to [batch, seq_len, d_model]
                    reshaped = activation.reshape(batch, seq_len, n_heads * d_head)
                    last_token_act = reshaped[0, -1, :].detach().cpu()
                elif len(activation.shape) >= 3:
                    # Take the last token's activation: [batch=1, seq_len, d_model] -> [d_model]
                    last_token_act = activation[0, -1, :].detach().cpu()
                elif len(activation.shape) == 2:
                    # Shape [seq_len, d_model] -> take last token
                    last_token_act = activation[-1, :].detach().cpu()
                else:
                    # Unexpected shape - try to reshape to d_model
                    flat = activation.view(-1).detach().cpu()
                    if len(flat) >= self.model.cfg.d_model:
                        last_token_act = flat[:self.model.cfg.d_model]
                    else:
                        # Pad if necessary
                        padding = torch.zeros(self.model.cfg.d_model - len(flat))
                        last_token_act = torch.cat([flat, padding])

                # Ensure we have the right size
                if len(last_token_act) != self.model.cfg.d_model:
                    if len(last_token_act) > self.model.cfg.d_model:
                        last_token_act = last_token_act[:self.model.cfg.d_model]
                    else:
                        padding = torch.zeros(self.model.cfg.d_model - len(last_token_act))
                        last_token_act = torch.cat([last_token_act, padding])

                self.activation_cache[hook_name] = last_token_act

            except Exception as e:
                print(f"Warning: Hook failed for {hook_name}: {e}, shape: {activation.shape}")
                # Store a zero vector as fallback
                self.activation_cache[hook_name] = torch.zeros(self.model.cfg.d_model)
            return activation
        return hook_fn

    def collect_activations(self, texts: List[str], labels: List[int]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Collect activations for a list of texts with better error handling"""
        all_activations = {name: [] for name in self.all_hook_names}
        all_labels = []

        # Process in smaller batches to avoid memory issues
        batch_size = min(8, len(texts))  # Small batch size for Colab

        for batch_start in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            batch_labels = labels[batch_start:batch_end]

            for idx, text in enumerate(batch_texts):
                self.activation_cache = {}

                # Tokenize and move to device
                tokens = self.model.to_tokens(text, prepend_bos=True).to(device)

                # Create hooks
                hooks = [(name, self._create_hook_fn(name)) for name in self.all_hook_names]

                # Run model with hooks
                try:
                    with torch.no_grad():
                        _ = self.model.run_with_hooks(tokens, fwd_hooks=hooks)

                    # Debug: Check what activations we actually got
                    if batch_start == 0 and idx == 0:  # Only for first example
                        captured_hooks = list(self.activation_cache.keys())
                        missing_hooks = [name for name in self.all_hook_names if name not in captured_hooks]
                        if captured_hooks:
                            print(f"✅ Captured activations: {len(captured_hooks)}/{len(self.all_hook_names)}")
                        if missing_hooks:
                            print(f"❌ Missing activations: {missing_hooks[:3]}...")

                    # Store activations
                    for layer_name in self.all_hook_names:
                        if layer_name in self.activation_cache:
                            all_activations[layer_name].append(self.activation_cache[layer_name])
                        else:
                            # Add zero vector if activation wasn't captured - but only warn once per layer
                            if layer_name not in all_activations or len(all_activations[layer_name]) == 0:
                                print(f"Warning: Missing activation for {layer_name}, using zeros")
                            all_activations[layer_name].append(torch.zeros(self.model.cfg.d_model))

                    all_labels.append(batch_labels[idx])

                except Exception as e:
                    print(f"Error processing text '{text[:50]}...': {e}")
                    continue

            # Clear memory after each batch
            clear_gpu_memory()

        # Convert to tensors, handling empty cases
        final_activations = {}
        for layer_name in all_activations:
            if all_activations[layer_name]:
                try:
                    final_activations[layer_name] = torch.stack(all_activations[layer_name])
                except Exception as e:
                    print(f"Error stacking activations for {layer_name}: {e}")
                    final_activations[layer_name] = torch.empty(0, self.model.cfg.d_model)
            else:
                final_activations[layer_name] = torch.empty(0, self.model.cfg.d_model)

        return final_activations, torch.tensor(all_labels, dtype=torch.long)

class BiasDetector:
    """Train probing classifiers to detect bias in activations with improved robustness"""

    def __init__(self, use_pca: bool = True, max_components: int = 128):
        self.probes = {}
        self.scalers = {}
        self.pcas = {}
        self.layer_importance = {}
        self.use_pca = use_pca
        self.max_components = max_components

    def train_probes(self, activations: Dict[str, torch.Tensor], labels: torch.Tensor):
        """Train linear probes for each layer with better preprocessing"""
        results = {}
        y = labels.numpy()

        # Check if we have both classes
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            print(f"Warning: Only found labels {unique_labels}. Need both 0 and 1 for binary classification.")
            print("Creating balanced synthetic data for demonstration...")

            # Add some synthetic examples to ensure we have both classes
            n_synthetic = max(4, len(y) // 4)  # At least 4 synthetic examples
            synthetic_labels = np.array([1 - y[0]] * n_synthetic)  # Opposite class
            y = np.concatenate([y, synthetic_labels])

            # Extend activations with synthetic data (slightly perturbed versions)
            for layer_name in activations:
                if activations[layer_name].shape[0] > 0:
                    # Create synthetic activations by adding noise to existing ones
                    base_acts = activations[layer_name]
                    noise = torch.randn_like(base_acts[:n_synthetic]) * 0.1
                    synthetic_acts = base_acts[:n_synthetic] + noise
                    activations[layer_name] = torch.cat([base_acts, synthetic_acts], dim=0)

        for layer_name, acts in activations.items():
            if acts.numel() == 0:
                print(f"Skipping empty layer: {layer_name}")
                continue

            print(f"Training probe for {layer_name} (shape: {acts.shape})")

            # Convert to numpy and ensure 2D shape
            X = acts.numpy()

            # Handle multi-dimensional activations
            if len(X.shape) > 2:
                print(f"  Reshaping {X.shape} -> {(X.shape[0], -1)}")
                X = X.reshape(X.shape[0], -1)
            elif len(X.shape) == 1:
                print(f"  Reshaping {X.shape} -> {(1, X.shape[0])}")
                X = X.reshape(1, -1)

            print(f"  Final shape: {X.shape}")

            # Handle the case where we have fewer samples than features
            if X.shape[0] < X.shape[1] and self.use_pca:
                print(f"  Using PCA: {X.shape[1]} -> {min(self.max_components, X.shape[0]-1)} dims")
                n_components = min(self.max_components, X.shape[0] - 1, X.shape[1])
                if n_components < 2:
                    print(f"  Skipping {layer_name}: insufficient data")
                    continue

            # Preprocessing pipeline
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Apply PCA if needed
            if self.use_pca and X_scaled.shape[1] > self.max_components:
                n_components = min(self.max_components, X_scaled.shape[0] - 1)
                pca = PCA(n_components=n_components, random_state=42)
                X_final = pca.fit_transform(X_scaled)
                self.pcas[layer_name] = pca
                print(f"  Applied PCA: {X_scaled.shape[1]} -> {X_final.shape[1]} dims")
            else:
                X_final = X_scaled

            # Split data with stratification
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_final, y, test_size=0.3, random_state=42, stratify=y
                )
            except ValueError:
                # If stratification fails, split without it
                print(f"  Warning: Stratification failed for {layer_name}")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_final, y, test_size=0.3, random_state=42
                )

            # Train probe with better regularization
            probe = LogisticRegression(
                random_state=42,
                max_iter=2000,
                class_weight='balanced',  # Handle class imbalance
                C=1.0  # Regularization strength
            )

            try:
                probe.fit(X_train, y_train)

                # Evaluate
                train_acc = probe.score(X_train, y_train)
                test_acc = probe.score(X_test, y_test)
                y_pred = probe.predict(X_test)

                # Calculate additional metrics
                try:
                    y_proba = probe.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
                except:
                    roc_auc = 0.5

                # Store trained components
                self.probes[layer_name] = probe
                self.scalers[layer_name] = scaler

                results[layer_name] = {
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'roc_auc': roc_auc,
                    'n_train': len(y_train),
                    'n_test': len(y_test)
                }

                print(f"  ✅ Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}, AUC: {roc_auc:.3f}")

            except Exception as e:
                print(f"  ❌ Failed to train probe for {layer_name}: {e}")
                continue

        # Calculate layer importance
        if results:
            self.analyze_layer_importance(results)

        return results

    def analyze_layer_importance(self, results: Dict) -> Dict[str, float]:
        """Analyze which layers are most important for bias detection"""
        layer_scores = {}

        for layer_name, metrics in results.items():
            # Combine test accuracy and AUC for importance score
            test_acc = metrics.get('test_accuracy', 0.5)
            roc_auc = metrics.get('roc_auc', 0.5)
            # Weighted average favoring test accuracy
            importance = 0.7 * test_acc + 0.3 * roc_auc
            layer_scores[layer_name] = importance

        # Sort by importance
        sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
        self.layer_importance = dict(sorted_layers)

        print("\n📊 Layer Importance Ranking:")
        for i, (layer, score) in enumerate(sorted_layers[:5]):
            print(f"  {i+1}. {layer}: {score:.3f}")

        return self.layer_importance

    def detect_bias(self, activations: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Detect bias in new activations using trained probes"""
        bias_scores = {}

        for layer_name, probe in self.probes.items():
            if layer_name not in activations:
                continue

            acts = activations[layer_name]
            if acts.numel() == 0:
                continue

            X = acts.numpy()
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            elif len(X.shape) > 2:
                # Reshape multi-dimensional activations to 2D
                X = X.reshape(X.shape[0], -1)

            # Apply same preprocessing as training
            try:
                scaler = self.scalers[layer_name]
                X_scaled = scaler.transform(X)

                # Apply PCA if it was used during training
                if layer_name in self.pcas:
                    X_scaled = self.pcas[layer_name].transform(X_scaled)

                # Get probability of bias class
                bias_prob = probe.predict_proba(X_scaled)[0, 1]
                bias_scores[layer_name] = float(bias_prob)

            except Exception as e:
                print(f"Warning: Failed to get bias score for {layer_name}: {e}")
                bias_scores[layer_name] = 0.5  # Neutral default

        return bias_scores

class SteeringVectorComputer:
    """Compute and apply steering vectors for bias mitigation with improved generation"""

    def __init__(self, model: HookedTransformer):
        self.model = model
        self.steering_vectors = {}

    def compute_steering_vectors(self, neutral_activations: Dict[str, torch.Tensor],
                                biased_activations: Dict[str, torch.Tensor]):
        """Compute steering vectors as difference between biased and neutral activations"""

        for layer_name in neutral_activations:
            if layer_name in biased_activations:
                neutral_acts = neutral_activations[layer_name]
                biased_acts = biased_activations[layer_name]

                # Skip if either is empty
                if neutral_acts.numel() == 0 or biased_acts.numel() == 0:
                    continue

                # Compute mean activations
                neutral_mean = neutral_acts.mean(dim=0)
                biased_mean = biased_acts.mean(dim=0)

                # Steering vector points from biased to neutral
                steering_vector = neutral_mean - biased_mean

                # Normalize the steering vector
                steering_norm = torch.norm(steering_vector)
                if steering_norm > 0:
                    steering_vector = steering_vector / steering_norm

                self.steering_vectors[layer_name] = steering_vector
                print(f"Computed steering vector for {layer_name} (norm: {steering_norm:.3f})")

        return self.steering_vectors

    def apply_steering_to_generation(self, prompt: str, layer_name: str, strength: float = 1.0,
                                   max_new_tokens: int = 20, temperature: float = 0.8) -> str:
        """Apply steering vector during text generation"""
        if layer_name not in self.steering_vectors:
            print(f"Warning: No steering vector for {layer_name}")
            return self._generate_baseline(prompt, max_new_tokens, temperature)

        steering_vector = self.steering_vectors[layer_name] * strength

        def steering_hook(activation, hook):
            # Apply steering to all tokens in the sequence
            if len(activation.shape) == 3:  # [batch, seq, hidden]
                batch_size, seq_len, hidden_dim = activation.shape
                steering_broadcast = steering_vector.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
                activation = activation + steering_broadcast.to(activation.device)
            return activation

        # Generate with steering
        input_tokens = self.model.to_tokens(prompt, prepend_bos=True).to(device)

        generated_tokens = input_tokens.clone()

        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.model.run_with_hooks(
                    generated_tokens,
                    fwd_hooks=[(layer_name, steering_hook)]
                )

                # Apply temperature and sample
                next_token_logits = logits[0, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)

                # Sample next token
                next_token = torch.multinomial(probs, 1)

                # Add to sequence
                generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)

                # Stop if we hit end token
                if next_token.item() == self.model.tokenizer.eos_token_id:
                    break

        # Decode and clean up
        generated_text = self.model.to_string(generated_tokens[0])

        # Remove the original prompt to get just the completion
        if generated_text.startswith(prompt):
            completion = generated_text[len(prompt):].strip()
        else:
            completion = generated_text.strip()

        return f"{prompt} {completion}"

    def _generate_baseline(self, prompt: str, max_new_tokens: int = 20, temperature: float = 0.8) -> str:
        """Generate text without steering (baseline)"""
        input_tokens = self.model.to_tokens(prompt, prepend_bos=True).to(device)
        generated_tokens = input_tokens.clone()

        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.model(generated_tokens)
                next_token_logits = logits[0, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)

                if next_token.item() == self.model.tokenizer.eos_token_id:
                    break

        generated_text = self.model.to_string(generated_tokens[0])
        if generated_text.startswith(prompt):
            completion = generated_text[len(prompt):].strip()
        else:
            completion = generated_text.strip()

        return f"{prompt} {completion}"

    def apply_steering(self, text: str, layer_name: str, strength: float = 1.0) -> str:
        """Legacy method - now redirects to generation method"""
        return self.apply_steering_to_generation(text, layer_name, strength)

class BiasVisualization:
    """Visualize bias detection results"""

    @staticmethod
    def plot_layer_importance(layer_importance: Dict[str, float]):
        """Plot layer importance for bias detection"""
        layers = list(layer_importance.keys())
        scores = list(layer_importance.values())

        # Extract layer numbers for better visualization
        layer_nums = [int(layer.split('.')[1]) for layer in layers if 'blocks' in layer]
        layer_scores = [score for layer, score in zip(layers, scores) if 'blocks' in layer]

        plt.figure(figsize=(12, 6))
        plt.plot(layer_nums, layer_scores, marker='o')
        plt.xlabel('Layer Number')
        plt.ylabel('Bias Detection Accuracy')
        plt.title('Bias Detection Accuracy by Layer')
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def plot_activation_pca(activations: Dict[str, torch.Tensor], labels: torch.Tensor,
                           layer_name: str):
        """Plot PCA visualization of activations"""
        if layer_name not in activations:
            return

        acts = activations[layer_name].numpy()

        # Apply PCA
        pca = PCA(n_components=2)
        acts_2d = pca.fit_transform(acts)

        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(acts_2d[:, 0], acts_2d[:, 1], c=labels.numpy(),
                            cmap='RdYlBu', alpha=0.7)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title(f'PCA Visualization of {layer_name}')
        plt.colorbar(scatter, label='Bias Label')
        plt.show()

class BiasAgent:
    """Main agent class with improved Colab compatibility and error handling"""

    def __init__(self, model_name: str = "gpt2-small"):
        print(f"🤖 Loading model: {model_name}")
        print("   This may take a moment and download ~500MB if first time...")

        try:
            self.model = HookedTransformer.from_pretrained(model_name, device=device)
            print(f"   ✅ Model loaded successfully!")
            print(f"   📊 Model info: {self.model.cfg.n_layers} layers, {self.model.cfg.d_model} hidden dim")

        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            print("   💡 Try using 'gpt2-small' or check your internet connection")
            raise

        self.collector = ActivationCollector(self.model)
        self.detector = BiasDetector()
        self.steering = SteeringVectorComputer(self.model)

        self.is_trained = False

        # Clear memory after model loading
        clear_gpu_memory()

    def train(self, dry_run: bool = False, save_path: Optional[str] = None):
        """Train the bias detection system with better error handling"""
        print("\n" + "="*60)
        print("🎯 TRAINING BIAS DETECTION SYSTEM")
        print("="*60)

        print("📝 Generating training dataset...")
        dataset = BiasDataset(neutral_only=dry_run)
        examples = dataset.generate_full_dataset()

        # Prepare data
        texts = [ex.text for ex in examples]
        labels = [ex.label for ex in examples]

        print(f"📊 Dataset stats:")
        print(f"   Total examples: {len(texts)}")
        print(f"   Neutral examples: {sum(1 for l in labels if l == 0)}")
        print(f"   Biased examples: {sum(1 for l in labels if l == 1)}")

        if dry_run:
            print("   🔹 Running in DRY-RUN mode (neutral examples only)")

        if len(texts) == 0:
            raise ValueError("No examples generated!")

        # Collect activations
        print("\n🧠 Collecting model activations...")
        try:
            activations, label_tensor = self.collector.collect_activations(texts, labels)
            print(f"   ✅ Collected activations for {len([k for k, v in activations.items() if v.numel() > 0])} layers")
        except Exception as e:
            print(f"   ❌ Failed to collect activations: {e}")
            raise

        # Train probes
        print("\n🔍 Training bias detection probes...")
        try:
            results = self.detector.train_probes(activations, label_tensor)
            print(f"   ✅ Trained {len(results)} probes successfully")
        except Exception as e:
            print(f"   ❌ Failed to train probes: {e}")
            raise

        # Compute steering vectors (only if we have both classes)
        print("\n🎯 Computing steering vectors...")
        try:
            neutral_mask = label_tensor == 0
            biased_mask = label_tensor == 1

            if neutral_mask.sum() > 0 and biased_mask.sum() > 0:
                neutral_acts = {k: v[neutral_mask] for k, v in activations.items() if v.numel() > 0}
                biased_acts = {k: v[biased_mask] for k, v in activations.items() if v.numel() > 0}

                steering_vectors = self.steering.compute_steering_vectors(neutral_acts, biased_acts)
                print(f"   ✅ Computed {len(steering_vectors)} steering vectors")
            else:
                print("   ⚠️  Skipping steering vectors (need both neutral and biased examples)")

        except Exception as e:
            print(f"   ❌ Failed to compute steering vectors: {e}")
            # Continue without steering vectors

        self.is_trained = True

        # Save model if path provided
        if save_path:
            try:
                self.save(save_path)
                print(f"   💾 Saved to {save_path}")
            except Exception as e:
                print(f"   ⚠️  Failed to save: {e}")

        # Visualizations (safe with try-catch)
        try:
            if self.detector.layer_importance:
                print("\n📈 Generating visualizations...")
                BiasVisualization.plot_layer_importance(self.detector.layer_importance)

                # PCA visualization for best layer
                best_layer = max(self.detector.layer_importance.items(), key=lambda x: x[1])[0]
                BiasVisualization.plot_activation_pca(activations, label_tensor, best_layer)
        except Exception as e:
            print(f"   ⚠️  Visualization failed: {e}")

        # Clear memory
        clear_gpu_memory()

        return results, self.detector.layer_importance

    def detect_bias_in_text(self, text: str) -> Dict[str, Any]:
        """Detect bias in a given text with better error handling"""
        if not self.is_trained:
            raise ValueError("❌ Agent must be trained before detecting bias")

        try:
            # Collect activations for this text
            activations, _ = self.collector.collect_activations([text], [0])

            # Detect bias
            bias_scores = self.detector.detect_bias(activations)

            # Aggregate score (weighted by layer importance)
            total_score = 0.0
            total_weight = 0.0

            for layer, score in bias_scores.items():
                weight = self.detector.layer_importance.get(layer, 0.0)
                total_score += score * weight
                total_weight += weight

            final_score = total_score / total_weight if total_weight > 0 else 0.5

            return {
                'overall_bias_score': float(final_score),
                'layer_scores': bias_scores,
                'is_biased': final_score > 0.5,
                'confidence': abs(final_score - 0.5) * 2  # How confident are we?
            }

        except Exception as e:
            print(f"⚠️  Error detecting bias in '{text[:30]}...': {e}")
            return {
                'overall_bias_score': 0.5,
                'layer_scores': {},
                'is_biased': False,
                'confidence': 0.0,
                'error': str(e)
            }

    def mitigate_bias(self, text: str, strength: float = 1.0, max_tokens: int = 20) -> Dict[str, str]:
        """Apply bias mitigation with comparison to baseline"""
        if not self.is_trained:
            raise ValueError("❌ Agent must be trained before mitigating bias")

        if not self.detector.layer_importance:
            print("⚠️  No layer importance available, using baseline generation")
            baseline = self.steering._generate_baseline(text, max_tokens)
            return {'original': text, 'baseline': baseline, 'mitigated': baseline}

        try:
            # Get best layer for steering
            best_layer = max(self.detector.layer_importance.items(), key=lambda x: x[1])[0]
            print(f"🎯 Using layer {best_layer} for steering (importance: {self.detector.layer_importance[best_layer]:.3f})")

            # Generate baseline and steered versions
            baseline = self.steering._generate_baseline(text, max_tokens)
            steered = self.steering.apply_steering_to_generation(text, best_layer, strength, max_tokens)

            return {
                'original': text,
                'baseline': baseline,
                'mitigated': steered,
                'steering_layer': best_layer,
                'steering_strength': strength
            }

        except Exception as e:
            print(f"⚠️  Error during mitigation: {e}")
            baseline = self.steering._generate_baseline(text, max_tokens)
            return {'original': text, 'baseline': baseline, 'mitigated': baseline, 'error': str(e)}

    def save(self, path: str):
        """Save trained components with better error handling"""
        try:
            save_data = {
                'probes': self.detector.probes,
                'scalers': self.detector.scalers,
                'pcas': self.detector.pcas,
                'layer_importance': self.detector.layer_importance,
                'steering_vectors': {k: v.cpu() for k, v in self.steering.steering_vectors.items()},
                'model_name': self.model.cfg.model_name if hasattr(self.model.cfg, 'model_name') else 'unknown'
            }

            with open(path, 'wb') as f:
                pickle.dump(save_data, f)

            print(f"💾 Model components saved to {path}")

        except Exception as e:
            print(f"❌ Failed to save model: {e}")

    def load(self, path: str):
        """Load trained components with better error handling"""
        try:
            with open(path, 'rb') as f:
                save_data = pickle.load(f)

            self.detector.probes = save_data.get('probes', {})
            self.detector.scalers = save_data.get('scalers', {})
            self.detector.pcas = save_data.get('pcas', {})
            self.detector.layer_importance = save_data.get('layer_importance', {})

            # Load steering vectors and ensure they're on CPU initially
            steering_data = save_data.get('steering_vectors', {})
            self.steering.steering_vectors = {
                k: v.cpu() if isinstance(v, torch.Tensor) else torch.tensor(v)
                for k, v in steering_data.items()
            }

            self.is_trained = True
            print(f"✅ Model components loaded from {path}")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

# Colab-optimized main function and utilities
def run_demo(dry_run: bool = False, model_name: str = "gpt2-small"):
    """Main demo function optimized for Colab"""
    print("🚀 Starting TransformerLens Bias Detection Demo")
    print("="*60)

    try:
        # Initialize agent
        agent = BiasAgent(model_name)

        # Train the system
        results, layer_importance = agent.train(dry_run=dry_run)

        # Test detection
        print("\n" + "="*60)
        print("🔍 TESTING BIAS DETECTION")
        print("="*60)

        test_texts = [
            "The nurse was gentle and caring",  # Neutral
            "The engineer solved the problem efficiently",  # Neutral
            "The doctor provided excellent care",  # Neutral
            "Employees of all generations bring diverse perspectives",  # Neutral - age
            "People of different faiths coexist peacefully",  # Neutral - religion
            "People with disabilities contribute valuable perspectives",  # Neutral - disability
            "People from different economic backgrounds deserve equal opportunities",  # Neutral - socioeconomic
        ]

        if not dry_run:
            # Add biased examples only if not in dry-run mode
            test_texts.extend([
                "Women are too emotional to be CEOs",  # Biased - gender
                "Men are naturally better at math than women",  # Biased - gender
                "Those people are naturally violent",  # Biased - race
                "Older workers are slow to learn new technology",  # Biased - age
                "People of that religion are fundamentally extremist",  # Biased - religion
                "Disabled people are a burden on society",  # Biased - disability
                "Poor people are lazy and irresponsible",  # Biased - socioeconomic
            ])

        for text in test_texts:
            result = agent.detect_bias_in_text(text)
            bias_emoji = "🚨" if result['is_biased'] else "✅"
            confidence_emoji = "🎯" if result.get('confidence', 0) > 0.3 else "🤔"

            print(f"{bias_emoji}{confidence_emoji} '{text}'")
            print(f"   Bias Score: {result['overall_bias_score']:.3f}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            print()

        # Test mitigation (only if we have steering vectors)
        if agent.steering.steering_vectors and not dry_run:
            print("="*60)
            print("🛡️  TESTING BIAS MITIGATION")
            print("="*60)

            test_prompts = [
                "Women are not suitable for",
                "Men should always be the ones who",
                "People from that group are naturally",
            ]

            for prompt in test_prompts:
                print(f"\n🔧 Prompt: '{prompt}'")
                try:
                    result = agent.mitigate_bias(prompt, strength=1.0, max_tokens=15)
                    print(f"   Baseline:  {result['baseline']}")
                    print(f"   Mitigated: {result['mitigated']}")

                    # Check if mitigation changed the output
                    if result['baseline'] != result['mitigated']:
                        print("   ✅ Steering had an effect!")
                    else:
                        print("   ℹ️  No change detected")

                except Exception as e:
                    print(f"   ❌ Mitigation failed: {e}")

        print("\n" + "="*60)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)

        # Print summary
        print(f"📊 Summary:")
        print(f"   Model: {model_name}")
        print(f"   Trained probes: {len(agent.detector.probes)}")
        print(f"   Steering vectors: {len(agent.steering.steering_vectors)}")
        print(f"   Best layer: {max(layer_importance.items(), key=lambda x: x[1])[0] if layer_importance else 'None'}")

        return agent

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Try running with dry_run=True for a safer test")
        raise

def run_interactive_demo(agent: BiasAgent):
    """Interactive demo for testing custom inputs"""
    print("\n🎮 Interactive Demo - Enter your own text to analyze!")
    print("Type 'quit' to exit, 'help' for commands")

    while True:
        try:
            user_input = input("\n📝 Enter text to analyze: ").strip()

            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'help':
                print("Commands:")
                print("  - Enter any text to get bias analysis")
                print("  - 'mitigate <text>' to test bias mitigation")
                print("  - 'quit' to exit")
                continue
            elif user_input.lower().startswith('mitigate '):
                prompt = user_input[9:]  # Remove 'mitigate '
                result = agent.mitigate_bias(prompt)
                print(f"🔧 Mitigation result:")
                print(f"   Original: {result['original']}")
                print(f"   Baseline: {result['baseline']}")
                print(f"   Mitigated: {result['mitigated']}")
                continue

            if not user_input:
                continue

            # Analyze the text
            result = agent.detect_bias_in_text(user_input)

            bias_emoji = "🚨" if result['is_biased'] else "✅"
            print(f"{bias_emoji} Analysis:")
            print(f"   Bias Score: {result['overall_bias_score']:.3f}")
            print(f"   Is Biased: {result['is_biased']}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    print("👋 Thanks for using the bias detection demo!")

# Example usage and testing
def main():
    """Main function demonstrating the bias detection system"""

    print("🔬 TRANSFORMERLENS BIAS DETECTION SYSTEM")
    print("="*60)
    print("This demo shows how to detect and mitigate bias using mechanistic interpretability")
    print("Optimized for Google Colab with improved error handling and memory management")
    print()

    # Run the main demo
    agent = run_demo(dry_run=False, model_name="gpt2-large")  # Set dry_run=True for safer testing

    # Optionally run interactive demo
    # Uncomment the next line if you want to test custom inputs
    # run_interactive_demo(agent)

# Colab-specific convenience functions
def quick_test():
    """Quick test function for Colab users"""
    print("🚀 Running quick bias detection test...")
    agent = run_demo(dry_run=True, model_name="gpt2-small")
    return agent

def full_demo():
    """Full demo with bias examples"""
    print("🚀 Running full bias detection demo...")
    agent = run_demo(dry_run=False, model_name="gpt2-small")
    return agent

if __name__ == "__main__":
    main()
