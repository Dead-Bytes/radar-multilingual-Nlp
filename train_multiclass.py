
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # --- 1. Load and Prepare the Dataset ---
    print("Loading the processed dataset...")
    try:
        df = pd.read_csv('processed_xai_dataset.csv')
    except FileNotFoundError:
        print("Error: 'processed_xai_dataset.csv' not found.")
        print("Please run 'prepare_data.py' first.")
        return

    # Create a mapping from string labels to integers
    labels = df['source_label'].unique()
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    df['label'] = df['source_label'].map(label2id)

    print("Dataset loaded. Class mapping:")
    print(label2id)

    # Split the dataset
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    # Convert pandas DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    dataset_dict = DatasetDict({'train': train_dataset, 'test': test_dataset})

    # --- 2. Model and Tokenizer Definition ---
    MODEL_NAME = "microsoft/mdeberta-v3-base" # A strong multilingual model
    print(f"\nLoading tokenizer for '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    print("Tokenizing the dataset...")
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

    # --- 3. Model Configuration ---
    print("Configuring the model for multi-class classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    # --- 4. Training Setup ---
    # Define compute_metrics function for evaluation
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # Define training arguments
    # Optimized for research-quality training
    training_args = TrainingArguments(
        output_dir='./results_multiclass',
        num_train_epochs=5,  # Increased for better convergence
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs_multiclass',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,  # Only keep best 2 checkpoints
        report_to="none",  # Disable wandb/tensorboard
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )

    print("\n" + "="*60)
    print("STARTING TRAINING - PHASE 2.2")
    print("="*60)

    # Train the model
    trainer.train()

    print("\n" + "="*60)
    print("TRAINING COMPLETE - EVALUATING MODEL")
    print("="*60)

    # --- 5. Evaluation and Visualization ---
    # Get predictions on test set
    predictions_output = trainer.predict(tokenized_datasets["test"])
    predictions = np.argmax(predictions_output.predictions, axis=1)
    true_labels = predictions_output.label_ids

    # Calculate per-class metrics
    print("\n--- Per-Class Performance Metrics ---")
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, labels=list(range(len(labels)))
    )

    metrics_df = pd.DataFrame({
        'Class': [id2label[i] for i in range(len(labels))],
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    print(metrics_df.to_string(index=False))

    # Overall metrics
    print("\n--- Overall Performance ---")
    overall_accuracy = accuracy_score(true_labels, predictions)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    print(f"Accuracy: {overall_accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")

    # Generate and save confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[id2label[i] for i in range(len(labels))],
        yticklabels=[id2label[i] for i in range(len(labels))]
    )
    plt.title('Confusion Matrix - Multi-Class Authorship Attribution', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix_multiclass.png', dpi=300, bbox_inches='tight')
    print("\n✓ Confusion matrix saved to 'confusion_matrix_multiclass.png'")

    # Save metrics to file
    metrics_df.to_csv('per_class_metrics.csv', index=False)
    print("✓ Per-class metrics saved to 'per_class_metrics.csv'")

    # Save overall metrics
    with open('overall_metrics.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("Multi-Class Authorship Attribution - Overall Performance\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Training Samples: {len(train_df)}\n")
        f.write(f"Test Samples: {len(test_df)}\n")
        f.write(f"Number of Classes: {len(labels)}\n\n")
        f.write(f"Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"Macro Precision: {macro_precision:.4f}\n")
        f.write(f"Macro Recall: {macro_recall:.4f}\n")
        f.write(f"Macro F1-Score: {macro_f1:.4f}\n\n")
        f.write("="*60 + "\n")
        f.write("Per-Class Metrics\n")
        f.write("="*60 + "\n\n")
        f.write(metrics_df.to_string(index=False))
    print("✓ Overall metrics saved to 'overall_metrics.txt'")

    # Save the final model
    trainer.save_model('./final_model_multiclass')
    print("✓ Final model saved to './final_model_multiclass'")

    print("\n" + "="*60)
    print("PHASE 2.2 COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
