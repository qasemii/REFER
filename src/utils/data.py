dataset_info = {
    'boolq': {
        'train': ['train', 6363],
        'dev': ['val', 1491],
        'test': ['test', 2807],
        'num_classes': 2,
        'classes': ['False', 'True'],
        'max_length': {
            'bert-base-uncased': 512,
            'google/bigbird-roberta-base': 4096,
        },
        'num_special_tokens': None,
    },
      'contrast_mnli_original': {
        'test': ['test', 1446],
        'num_classes': 3,
        'classes': ['entailment', 'neutral', 'contradiction'],
        'max_length': {
            'bert-base-uncased': 250,
            'google/bigbird-roberta-base': 250,
        },
        'num_special_tokens': 3,
    },
    'contrast_mnli_contrast': {
        'test': ['test', 4285],
        'num_classes': 3,
        'classes': ['entailment', 'neutral', 'contradiction'],
        'max_length': {
            'bert-base-uncased': 250,
            'google/bigbird-roberta-base': 250,
        },
        'num_special_tokens': 3,
    },
    'cose': {
        'train': ['train', 8752],
        'dev': ['val', 1086],
        'test': ['test', 1079],
        'num_classes': 5,
        'classes': ['A', 'B', 'C', 'D', 'E'],
        'max_length': {
            'bert-base-uncased': 512,
            'google/bigbird-roberta-base': 77,
        },
        'num_special_tokens': None,
    },
    'esnli': {
        'train': ['train', 549309],
        'dev': ['val', 9823],
        'test': ['test', 9807],
        'num_classes': 3,
        'classes': ['entailment', 'neutral', 'contradiction'],
        'max_length': {
            'bert-base-uncased': 125,
            'google/bigbird-roberta-base': 125,
        },
        'num_special_tokens': 3,
    },
    'mnli': {
        'train': ['train', 10000],
        'dev': ['validation_matched', 2000],
        'test': ['validation_matched', 2000],
        'num_classes': 3,
        'classes': ['entailment', 'neutral', 'contradiction'],
        'max_length': {
            'bert-base-uncased': 250,
            'google/bigbird-roberta-base': 250,
        },
        'num_special_tokens': 3,
    },
    'emnli': {
        'train': ['train', 1],
        'num_classes': 3,
        'classes': ['entailment', 'neutral', 'contradiction'],
        'max_length': {
            'bert-base-uncased': 250,
            'google/bigbird-roberta-base': 250,
        },
        'num_special_tokens': 3,
    },
    'hans': {
        'train': ['train', 30000],
        'dev': ['validation', 15000],
        'test': ['validation', 15000],
        'num_classes': 2,
        'classes': ['entailment', 'non-entailment'],
        'max_length': {
            'bert-base-uncased': 250,
            'google/bigbird-roberta-base': 250,
        },
        'num_special_tokens': 3,
    },
    'evidence_inference': {
        'train': ['train', 7958],
        'dev': ['val_exhaustive', 1073],
        'test': ['test_exhaustive', 1111],
        'num_classes': 3,
        'classes': ['significantly increased', 'significantly decreased', 'no significant difference'],
        'max_length': {
            'bert-base-uncased': 512,
            'google/bigbird-roberta-base': 4096,
        },
        'num_special_tokens': None,
    },
    'fever': {
        'train': ['train', 97957],
        'dev': ['val', 6122],
        'test': ['test', 6111],
        'num_classes': 2,
        'classes': ['REFUTES', 'SUPPORTS'],
        'max_length': {
            'bert-base-uncased': 512,
            'google/bigbird-roberta-base': 1024,
        },
        'num_special_tokens': 3,
    },
    'movies': {
        'train': ['train', 1599],
        'dev': ['val', 200],
        'test': ['test', 200],
        'num_classes': 2,
        'classes': ['NEG', 'POS'],
        'max_length': {
            'bert-base-uncased': 512,
            'google/bigbird-roberta-base': 1024,
        },
        'num_special_tokens': 2,
    },
    'multirc': {
        'train': ['train', 24029],
        'dev': ['val', 3214],
        'test': ['test', 4848],
        'num_classes': 2,
        'classes': ['False', 'True'],
        'max_length': {
            'bert-base-uncased': 512,
            'google/bigbird-roberta-base': 748,
        },
        'num_special_tokens': 3,
    },
    'scifact': {
        'train': ['train', 405],
        'dev': ['validation', 100],
        'test': ['test', 188],
        'num_classes': 2,
        'classes': ['REFUTES', 'SUPPORTS'],
        'max_length': {
            'bert-base-uncased': 512,
            'google/bigbird-roberta-base': 4096,
        },
        'num_special_tokens': None,
    },
    'sst': {
        'train': ['train', 6920],
        'dev': ['dev', 872],
        'test': ['test', 1821],
        'num_classes': 2,
        'classes': ['neg', 'pos'],
        'max_length': {
            'bert-base-uncased': 58,
            'google/bigbird-roberta-base': 67,
        },
        'num_special_tokens': 2,
    },
    'amazon': {
        'train': ['train', 10000],
        'dev': ['dev', 2000],
        'test': ['test', 2000],
        'num_classes': 2,
        'classes': ['neg', 'pos'],
        'max_length': {
            'bert-base-uncased': 256,
            'google/bigbird-roberta-base': 256,
        },
        'num_special_tokens': 2,
    },
    'yelp': {
        'train': ['train', 10000],
        'dev': ['dev', 2000],
        'test': ['test', 2000],
        'num_classes': 2,
        'classes': ['neg', 'pos'],
        'max_length': {
            'bert-base-uncased': 512,
            'google/bigbird-roberta-base': 512,
        },
        'num_special_tokens': 2,
    },
    'hatexplain': {
        'train': ['train', 15383],
        'dev': ['val', 1922],
        'test': ['test', 1924],
        'num_classes': 2,
        'classes': ['non-toxic', 'toxic'],
        'max_length': {
            'bert-base-uncased': 300,
            'google/bigbird-roberta-base': 850,
        },
        'num_special_tokens': 2,
    },
    'checklist_flight': {
        'test': ['test', 87469],
        'num_classes': 2,
        'classes': ['neg', 'pos'],
        'max_length': {
            'bert-base-uncased': 58,
            'google/bigbird-roberta-base': 67,
        },
        'num_special_tokens': 2,
    },
    'imdb':{
        'train': ['train', ],
        'dev': ['dev', 100],
        'test': ['test', 100],
        'num_classes': 2,
        'classes': ['neg', 'pos'],
        'max_length': {
            'bert-base-uncased': 512,
            'google/bigbird-roberta-base': 512,
        },
        'num_special_tokens': 2,
    },
    'checklist_snli': {
        'test': ['test', 2614],
        'num_classes': 3,
        'classes': ['entailment', 'neutral', 'contradiction'],
        'max_length': {
            'bert-base-uncased': 250,
            'google/bigbird-roberta-base': 250,
        },
        'num_special_tokens': 3,
    },
    'checklist_sst': {
        'test': ['test', 29756],
        'num_classes': 2,
        'classes': ['neg', 'pos'],
        'max_length': {
            'bert-base-uncased': 58,
            'google/bigbird-roberta-base': 67,
        },
        'num_special_tokens': 2,
    },
}

eraser_datasets = ['boolq', 'cose', 'esnli', 'evidence_inference', 'fever', 'movies', 'multirc', 'scifact']

monitor_dict = {
    'boolq': 'dev_macro_f1_metric_epoch',
    'cose': 'dev_acc_metric_epoch',

    'mnli':'dev_macro_f1_metric_epoch',
    'hans': 'dev_macro_f1_metric_epoch',

    'esnli': 'dev_loss_epoch',
    'emnli': 'dev_loss_epoch',

    'evidence_inference': 'dev_macro_f1_metric_epoch',
    'fever': 'dev_macro_f1_metric_epoch',
    'movies': 'dev_macro_f1_metric_epoch',
    'multirc': 'dev_macro_f1_metric_epoch',
    'scifact': 'dev_macro_f1_metric_epoch',
    'sst': 'dev_loss_epoch',
    'amazon': 'dev_acc_metric_epoch',
    'yelp': 'dev_acc_metric_epoch',
    'hatexplain':'dev_acc_metric_epoch',
    'checklist_flight': 'dev_acc_metric_epoch',
    'imdb':'dev_acc_metric_epoch',
    'checklist_snli':'dev_macro_f1_metric_epoch',
    'checklist_sst': 'dev_acc_metric_epoch',
    'contrast_mnli_original':'dev_macro_f1_metric_epoch',
    'contrast_mnli_contrast':'dev_macro_f1_metric_epoch',
}

data_keys = ['item_idx', 'input_ids', 'attention_mask', 'rationale', 'inv_rationale', 'rand_rationale', 'has_rationale', 'label', 'rationale_indices']