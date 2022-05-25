# Grounded Compositional Generalization Research

This repository contains code and links to datasets for
Compositional Generalization research at the
Aalto Artificial Intelligence and Deep Learning Research Group (Aalto AI).

## Compositional Generalization via Induced Model Sparsity (NAACL-SRW 2022)

This work demonstrates a model that promotes sparse interactions between
words and disentangled factors and lead to better compositional generalization
performance at a low number of samples on a simple goal-reaching task.

## Generating the Dataset

The dataset we use for the paper in HDF5 can be found [here](https://aalto-naacl-2022-sparse-compgen.s3.us-west-1.amazonaws.com/archives/data.tar.xz).

The data is procedurally generated using the process described in Appendix B. We
collect 10,000 demonstrations of each trajectory, then split them into in-distribution
and out-of-distribution goals.

To generate examples with the bot over many threads, use:

    python scripts/generation/babyai/generate_examples_bot.py --help
    usage: generate_examples_bot.py [-h]
                                    [--n-trajectories-per-goal N_TRAJECTORIES_PER_GOAL]
                                    [--min-trajectory-length MIN_TRAJECTORY_LENGTH]
                                    [--seeds-and-solutions-buffer SEEDS_AND_SOLUTIONS_BUFFER]
                                    [--env ENV] [--n-goals N_GOALS]
                                    [--n-procs N_PROCS]
                                    output

    positional arguments:
      output                Where to store the resulting bot trajectories

    optional arguments:
      -h, --help            show this help message and exit
      --n-trajectories-per-goal N_TRAJECTORIES_PER_GOAL
      --min-trajectory-length MIN_TRAJECTORY_LENGTH
      --seeds-and-solutions-buffer SEEDS_AND_SOLUTIONS_BUFFER
      --env ENV             Name of environment to use
      --n-goals N_GOALS     Number of unique goals in this environment
      --n-procs N_PROCS

For example, our invocation looks something like:

    python scripts/generation/babyai/generate_examples_bot.py bot_trajectories_10000_each_BabyAI-GoToLocal-v0.pt --n-trajectories-per-goal 10000 --min-trajectory-length 7 --seeds-and-solutions-buffer seends_and_solutions_10000_each_BabyAI-GoToLocal-v0.pt --env BabyAI-GoToLocal-v0 --n-goals 36 --n-procs 8


This will generate the seeds, solve each trajectory and then replay the solutions to get the observations. Note that this
process can use quite a bit of memory, since it writes everything to a pickle file at the end.

Once the bot trajectories are generated you'll want to split them into training and OOD validation sets, then pad them so that they can be used with pytorch. This pads them so
that each trajectory is the length of the longest trajectory. Do the padding with:

    python scripts/generation/babyai/preprocess_data_bot.py --help 

    usage: preprocess_data_bot.py [-h] --bot-trajectories BOT_TRAJECTORIES output

    positional arguments:
      output

    optional arguments:
      -h, --help            show this help message and exit
      --bot-trajectories BOT_TRAJECTORIES

If you want to convert the pickle into something a little more portable, you can use the `scripts/generation/babyai/dataset_to_portable.py` script. This
converts the pickle file into an hdf5 file.

    python scripts/generation/babyai/dataset_to_portable.py --help 
    usage: dataset_to_portable.py [-h] pickle_file hdf5_file

    positional arguments:
      pickle_file
      hdf5_file

    optional arguments:
      -h, --help   show this help message and exit

### Contributions

The key contributions of the work (and their corresponding locations):

 - A form of factored attention (operating over factored representations) plus sparsity penalty (Section 4.1)
   - Factored embedding `models/interaction/independent_attention:IndependentAttentionModel:forward`. This uses the provided `attrib_offsets` to index into a single embedding, partitioned by the offsets into embeddings for each factor. Note that there is a separate channel for each of the factor embeddings - they are not
   concatenated together.
   - Factored attention `models/interaction/independent_attention:IndependentAttentionModel:forward`. Attention between the words and each cell in the observation state is computed over every factor separately, then multiplied together (using exp-sum-log). See equation (1) in the paper.
   - Sparsity penalty between outer product of word embedding space and attribute embedding space `models/discriminator/independent_attention:IndependentAttentionProjectionHarness:training_step`. This is computed as logged as `l1c` and added to the loss from the original harness training step.
  - Discriminator training method to learn to find goals in unlabelled observations (Section 4.2).
    - The "Masking Module" is found in `models/img_mask:ImageComponentsToMask`
    - The training step for the discriminator is found in `models/discriminator/harness:ImageDiscriminatorHarness`. Here both $\mathcal{L}_{\text{img}}$ and $\mathcal{L}_{\text{int}}$ are computed.
    - The sampling and dataloading procedure are found in `datasets/discriminator:DiscriminatorDataset`. Here the sampling procedure described in Appendix D is implemented.
  - Planning Module
    - MVProp network `models/vin/mvprop:MVProp2D`
    - Q-value convolutional net `models/vin/resnet_value:BigConvolutionValueMapExtractor`
    - Conservative Q Learning-like penalty `models/vin/harness:VINHarness.training_step`

## Baselines

The baselines used in the paper for the sample efficiency
test can be found in `models/imitation`

|Name            |Module         |Description     |
|:--------------:|:-------------:|:--------------:|
|FiLM/GRU        |`baseline`     | Encode the instruction with LSTM and FiLM-modulate convolutional processing of observations, with global spatial max pooling  |
|Transformer     |`transformer`  | Encode the instruction using a position-encoded transformer encoder, decode the image using cross-attention between encoded instruction and 2D-position image, predicting the policy from a final CLS token|

## Ablations of our model

In the paper the following ablation of the model is reported:

| Name            | Experiment Name | Description     |
|:---------------:|:---------------:|:---------------:|
| MVProp/CNN      | `vin_sample_nogoal_16_logsumexp_k_0:mvprop` | Do not perform any value iteration steps, instead immediately concatenate the result of the interaction network to the observation and regress Q-values from there |

Other ablations not reported in the paper but available in
the code:

| Name            | Experiment Name | Description     |
|:---------------:|:---------------:|:---------------:|
| MVProp/Transformer      | `vin_sample_nogoal_16_logsumexp_ignorance_transformer:mvprop` | Use pre-trained Transformers (from the discriminator) to detect goals as opposed to the sparse factored attention  |

## Running the training process

There are three different things to train in order to reproduce the results in the paper.

For all three training script, logging happens to tensorboard files in the
directory given by `--experiment` / `--model`. You can run tensorboard
in that directory to see the results as the training is running. We also use
the tensorboard log files later on for analysis.

### Training the imitation learning baselines

    python scripts/train/train_imitation.py --help
    usage: train_imitation.py [-h] --data DATA --exp-name EXP_NAME --seed SEED
                              [--model {film_lstm_policy,conv_transformer,sentence_encoder_image_decoder,fused_inputs_next_step_encoder,fused_inputs_autoregressive_transformer}]
                              [--limit LIMIT] [--vlimit VLIMIT] [--tlimit TLIMIT]
                              [--total TOTAL] [--iterations ITERATIONS]
                              [--n-eval-procs N_EVAL_PROCS]
                              [--batch-size BATCH_SIZE]
                              [--check-val-every CHECK_VAL_EVERY]

    optional arguments:
      -h, --help            show this help message and exit
      --data DATA
      --exp-name EXP_NAME
      --seed SEED
      --model {film_lstm_policy,conv_transformer,sentence_encoder_image_decoder,fused_inputs_next_step_encoder,fused_inputs_autoregressive_transformer}
      --limit LIMIT
      --vlimit VLIMIT
      --tlimit TLIMIT
      --total TOTAL         Total number of instances per task
      --iterations ITERATIONS
      --n-eval-procs N_EVAL_PROCS
                            Number of processes to run evaluation with
      --batch-size BATCH_SIZE
                            Batch size for training
      --check-val-every CHECK_VAL_EVERY
                            Check val every N steps

The main thing that changes here is `--model`. The available models are:

|Name            |Module         |Description     |
|:--------------:|:-------------:|:--------------:|
|FiLM/GRU        |`film_lstm_policy`     | LSTM/FiLM  |
|Transformer     |`sentence_encoder_image_decoder`  | Transformer Encoder-Decoder (sometimes referred to as `pure_transformer`) |
|Next-step Encoder Transformer |`fused_inputs_next_step_encoder`  | Transformer Encoder-only model (not used in the paper) |
|Autoregressive Encoder Transformer |`fused_inputs_next_step_encoder`  | Transformer Encoder-only model which predicts all future steps from single frame (not used in the paper) |

For the paper we used the following invocation:

    python scripts/train/train_imitation.py --exp-name imitation --model $MODEL  --limit $LIMIT --vlimit 20 --tlimit 40 --iterations 120000 --data $DATA_PATH --seed $SEED


### Training the different discriminators

    python scripts/train/train_discriminator.py --help
    usage: train_discriminator.py [-h] --data DATA --exp-name EXP_NAME --seed SEED
                                  [--model {film,transformer,independent,independent_noreg,simple,simple_noreg}]
                                  [--limit LIMIT] [--vlimit VLIMIT]
                                  [--tlimit TLIMIT] [--iterations ITERATIONS]
                                  [--total TOTAL] [--batch-size BATCH_SIZE]
                                  [--check-val-every CHECK_VAL_EVERY]

    optional arguments:
      -h, --help            show this help message and exit
      --data DATA
      --exp-name EXP_NAME
      --seed SEED
      --model {film,transformer,independent,independent_noreg,simple,simple_noreg}
      --limit LIMIT         Training set limit (per task)
      --vlimit VLIMIT       Validation set limit (per task)
      --tlimit TLIMIT       Test set limit (per task)
      --iterations ITERATIONS
      --total TOTAL         Total number of instances per task
      --batch-size BATCH_SIZE
                            Batch size for training
      --check-val-every CHECK_VAL_EVERY
                            Check val every N steps

The main thing that changes here is `--model`. The available models are:

|Name            |Module         |Description     |
|:--------------:|:-------------:|:--------------:|
|FiLM/GRU        |`film`         | LSTM/FiLM  |
|Transformer     |`transformer`  | Transformer Encoder-Decoder |
|Sparse Factored Attention |`independent` | Our "Factored Attention" model with sparsity |
|Factored Attention |`independent_noreg` | Our "Factored Attention" model without sparsity |
|Sparse Attention |`simple`        | Non-factored attention where the input factor embeddings are concatenated together and projected into the word embedding space, with sparsity |

For the paper we used the following invocation:

    python scripts/train/train_discriminator.py --exp-name discriminator_sample_nogoal_16_logsumexp --model $MODEL  --limit $LIMIT --vlimit 20 --tlimit 40 --iterations $ITERATIONS --data $DATA_PATH --seed $SEED --batch-size 1024

### Training the MVProp module with the discriminator

    python scripts/train/train_vin.py --help 
    usage: train_vin.py [-h] --data DATA --exp-name EXP_NAME --seed SEED
                        [--model {mvprop}] [--limit LIMIT] [--vlimit VLIMIT]
                        [--tlimit TLIMIT] [--total TOTAL]
                        [--iterations ITERATIONS] [--n-eval-procs N_EVAL_PROCS]
                        [--batch-size BATCH_SIZE]
                        [--check-val-every CHECK_VAL_EVERY]
                        [--interaction-model {independent,transformer}]
                        [--load-interaction-model LOAD_INTERACTION_MODEL]
                        [--vin-k VIN_K] [--device DEVICE] [--show-progress]

    optional arguments:
      -h, --help            show this help message and exit
      --data DATA
      --exp-name EXP_NAME
      --seed SEED
      --model {mvprop}
      --limit LIMIT
      --vlimit VLIMIT
      --tlimit TLIMIT
      --total TOTAL         Total number of instances per task
      --iterations ITERATIONS
      --n-eval-procs N_EVAL_PROCS
                            Number of processes to run evaluation with
      --batch-size BATCH_SIZE
                            Batch size for training
      --check-val-every CHECK_VAL_EVERY
                            Check val every N steps
      --interaction-model {independent,transformer}
                            Which interaction module to use
      --load-interaction-model LOAD_INTERACTION_MODEL
                            Path to an interaction model to load
      --vin-k VIN_K         Number of VIN iterations
      --device DEVICE       Which device to use
      --show-progress       Show the progress bar


The main thing that changes here is `--interaction-model`. The available models are:

|Name            |Module         |Description     |
|:--------------:|:-------------:|:--------------:|
|Transformer     |`transformer`  | Transformer Encoder-Decoder |
|Sparse Factored Attention |`independent` | Our "Factored Attention" model with sparsity |

The appropriate value of `--vin-k` depends on the size of the observation space. You should set
it so that it is possible for a value from one end of the observation to propgate to the other end
if necessary, so that state-values can be correctly estimated. Since the kernel size for value propagation
is 3, you can use $\frac{\sqrt{H^2 + W^2}}{2} + 1$.

When training this model, you can load a pre-trained interaction model used for goal-detection with
`--load-interaction-model` and specifying a path to the saved weights.

For the paper we used the following invocation:

    BASE_MODEL=$DISCRIMINATOR_MODELS_PATH/discriminator_sample_nogoal_16_logsumexp/${INTERACTION_MODEL_NAME}/discriminator_sample_nogoal_16_logsumexp_s_${SEED}_m_${INTERACTION_MODEL_NAME}_it_200000_b_1024_l_${LIMIT}.pt    

    python scripts/train_vin.py --exp-name vin_sample_nogoal_16_logsumexp_ignorance_${INTERACTION_MODEL_NAME} --model $MODEL  --limit $LIMIT --vlimit 20 --tlimit 40 --iterations $ITERATIONS --data $DATA_PATH --seed $SEED --interaction-model ${INTERACTION_MODEL_NAME} --load-interaction-model $BASE_MODEL

## Postprocessing the results

Once the experimental runs are in, you should postprocess the checkpoints to prune out optimizer information.

Use the `convert_checkpoints.py` script for that.

    python scripts/postprocessing/convert_checkpoints.py --help 
    usage: convert_checkpoints.py [-h] [--dry-run] [--pattern PATTERN] directory

    positional arguments:
      directory

    optional arguments:
      -h, --help         show this help message and exit
      --dry-run
      --pattern PATTERN

This converts all the checkpoints in a directory (recursively) such that the optimizer information is removed.

## Analyzing the data and generating the plots

Model checkpoints and logs we generated by training on our GPU cluster can be found [here](https://aalto-naacl-2022-sparse-compgen.s3.us-west-1.amazonaws.com/archives/results.tar.bz2)

Now that you have the checkpoints, its time to do the analysis and reproduce the results in the paper!

### Sample efficiency curves - Figure 3

These are generated using the `rliable` package. First we need to extract the sample efficiency information from the logs:

    python scripts/analysis/babyai/extract_sample_efficiency_curves.py --help
    usage: extract_sample_efficiency_curves.py [-h]
                                              [--limits [LIMITS [LIMITS ...]]]
                                              [--check-experiments [CHECK_EXPERIMENTS [CHECK_EXPERIMENTS ...]]]
                                              [--check-models]
                                              [--n-procs N_PROCS]
                                              logs_dir models_dir output_file

    positional arguments:
      logs_dir
      models_dir
      output_file

    optional arguments:
      -h, --help            show this help message and exit
      --limits [LIMITS [LIMITS ...]]
      --check-experiments [CHECK_EXPERIMENTS [CHECK_EXPERIMENTS ...]]
                            Experiments in the format
                            experiment:model:iterations:cut_at
      --check-models        Whether to check if the models exist
      --n-procs N_PROCS


You can do this over multiple processes in order to speed it up a bit. The script only needs
the logs directory and models directory. An example invocation might be:

    python scripts/analysis/babyai/extract_sample_efficiency_curves.py $LOGS_PATH $MODELS_PATH sample_efficiency_curves.json

Then to plot the curves (what you see in the figure) you can use

    python scripts/analysis/babyai/plot_sample_efficiency_curves.py sample_efficiency_curves.json id.pdf ood.pdf

This creates two files containing the in-distribution and out-of-distribution curves, respectively.

### Predictions from the interaction models - Figure 4a

First we need to extract the predictions:

    python scripts/analysis/babyai/extract_sample_predictions.py --help
    usage: extract_sample_predictions.py [-h] [--models [MODELS [MODELS ...]]]
                                        [--limits [LIMITS [LIMITS ...]]]
                                        [--experiment EXPERIMENT]
                                        [--train-index TRAIN_INDEX]
                                        [--valid-index VALID_INDEX]
                                        dataset logs_dir models_dir output_json

    positional arguments:
      dataset
      logs_dir
      models_dir
      output_json

    optional arguments:
      -h, --help            show this help message and exit
      --models [MODELS [MODELS ...]]
      --limits [LIMITS [LIMITS ...]]
      --experiment EXPERIMENT
      --train-index TRAIN_INDEX
      --valid-index VALID_INDEX


The main arugments to pay attention to here are `--train-index` and `--valid-index` as they specify
which sample in the dataset should be used for the visualization.

An example invocation might be:

    python scripts/analysis/babyai/extract_sample_predictions.py $DATASET_PATH $LOGS_PATH $MODELS_PATH sample_predictions.json --train-index 0 --valid-index 3

Once that's done you can plot the predictions.

    python scripts/analysis/babyai/plot_sample_predictions.py sample_predictions.json sample_predictions.pdf

### Internal correlations - Figure 4b

First we need to extract the embedding dot products

    python scripts/analysis/babyai/extract_embedding_dot_products.py --help
    usage: extract_embedding_dot_products.py [-h] [--experiment EXPERIMENT]
                                            [--models [MODELS [MODELS ...]]]
                                            [--limits [LIMITS [LIMITS ...]]]
                                            logs_dir models_dir output_json

    positional arguments:
      logs_dir
      models_dir
      output_json

    optional arguments:
      -h, --help            show this help message and exit
      --experiment EXPERIMENT
      --models [MODELS [MODELS ...]]
      --limits [LIMITS [LIMITS ...]]
  
An example invocation might be:

    python scripts/analysis/babyai/extract_embedding_dot_products.py $LOGS_PATH $MODELS_PATH internal_correlations.json

Once that's done you can plot them:

    python scripts/analysis/babyai/plot_embedding_dot_products.py internal_correlations.json internal_correlations.pdf
