# Very Deep Convolutional Neural Network For Text Classification #

## Usage ##
```
python __main__.py  [-h] [--n_sentiment_classes N_SENTIMENT_CLASSES]
	                   [--n_comment_classes N_COMMENT_CLASSES]
	                   [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
	                   [--input_data INPUT_DATA] [--test_data TEST_DATA]
	                   [--output_dir OUTPUT_DIR] [--streamer STREAMER]
	                   [--embedding_size EMBEDDING_SIZE]
	                   [--max_feature_length MAX_FEATURE_LENGTH]
	                   [--evaluate_every EVALUATE_EVERY]
	                   [--checkpoint_every CHECKPOINT_EVERY]
	                   [--l2_weight_decay L2_WEIGHT_DECAY] [--train] [--infer]

optional arguments:
  -h, --help            show this help message and exit
  --n_sentiment_classes N_SENTIMENT_CLASSES
                        Specify number of classes of sentiments
  --n_comment_classes N_COMMENT_CLASSES
                        Specify number of classes of comments
  --num_epochs NUM_EPOCHS
                        Specify number of classes
  --batch_size BATCH_SIZE
                        Batch size. Must divide evenly into the dataset sizes.
  --input_data INPUT_DATA
                        Location store the input data (only accept `csv`
                        format)
  --test_data TEST_DATA
                        Specify test data path
  --output_dir OUTPUT_DIR
                        Directory to store the summaries and checkpoints.
  --streamer STREAMER   Specify a twitch streamer
  --embedding_size EMBEDDING_SIZE
                        Specify embedding size
  --max_feature_length MAX_FEATURE_LENGTH
                        Specify max feature length
  --evaluate_every EVALUATE_EVERY
                        do evaluation after # numbers of training steps
  --checkpoint_every CHECKPOINT_EVERY
                        Save checkpoint after # numbers of training steps
  --l2_weight_decay L2_WEIGHT_DECAY
                        Specify max feature length
  --train               Specify mode: training
  --infer               Specify mode: inferring
```


