import configparser

import datasets
import evaluate
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments)
import logging

logger = logging.getLogger(__name__)


log_dir = Path("logs/")
if not log_dir.exists():
    log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=log_dir /
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.log",
                    format='%(asctime)s %(name)-12s '
                    '%(levelname)-8s %(message)s',
                    filemode='w',
                    level=logging.INFO)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_labels = [[label_list[l] for l in label if l != -100]
                   for label in labels]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(
        predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("experiment.ini")
    logger.info('Experimnet with config: {}'.format(config))

    label_list = ['O', 'B-SKILL', 'I-SKILL',
                  'B-TITLE', 'I-TITLE', 'B-SALARY', 'I-SALARY']
    label2id = {'O': 0, 'B-SKILL': 1, 'I-SKILL': 2,
                'B-TITLE': 3, 'I-TITLE': 4, 'B-SALARY': 5, 'I-SALARY': 6}

    id2label = {v: k for k, v in label2id.items()}

    features = datasets.Features(
        {
            "id": datasets.Value("string"),
            "tokens": datasets.Sequence(datasets.Value("string")),
            "ner_tags": datasets.Sequence(
                datasets.features.ClassLabel(
                    names=label_list
                )
            ),
        }
    )

    data_files = {
        "train": config['TRAIN']["train_data_path"],
        "validation": config['TRAIN']["val_data_path"],
        "test": config['TEST']["test_data_path"]
    }

    dataset = datasets.load_dataset(
        "json", data_files=data_files, features=features)

    tokenizer = AutoTokenizer.from_pretrained(config['TRAIN']["model_name"])

    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    metric = evaluate.load("seqeval")

    model = AutoModelForTokenClassification.from_pretrained(
        config['TRAIN']["model_name"],
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=config['TRAIN']["exp_name"],
        learning_rate=config.getfloat('TRAIN', 'lr'),
        per_device_train_batch_size=config.getint("TRAIN", "batch_size"),
        per_device_eval_batch_size=config.getint("TRAIN", "batch_size"),
        num_train_epochs=config.getint("TRAIN", "epochs"),
        weight_decay=config.getfloat('TRAIN', 'weight_decay'),
        gradient_accumulation_steps=config.getint(
            "TRAIN", "accumulation_steps"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=config.getint("TRAIN", "logging_steps"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    logger.info("Training started")
    trainer.train()

    test_result = trainer.predict(tokenized_dataset['test']).metrics
    logger.info(f"Test results: {test_result}")
