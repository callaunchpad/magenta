
convert_dir_to_note_sequences \
	--input_dir=/Users/arshzahed/dev/launchpad/lmd_matched/A/A/A\
 	--output_file=/Users/arshzahed/dev/launchpad/datasets/datasets/test/lmd_matched.tfrecord  \
 	--recursive

melody_rnn_create_dataset \
--config='phase_rnn' \
--input=/Users/arshzahed/dev/launchpad/datasets/datasets/test/phase_try2.tfrecord \
--output_dir=/Users/arshzahed/dev/launchpad/datasets/datasets/test/phase_try2 \
--eval_ratio=0.10

python melody_rnn_create_dataset.py --melody_encoder_decoder='key' --input=/Users/arshzahed/dev/launchpad/datasets/datasets/lmd_matched.tfrecord --output_dir=/Users/arshzahed/dev/launchpad/datasets/datasets/test/phase_try_1 --eval_ratio=0.10

melody_rnn_train \
--config="phase_rnn" \
--run_dir=/Users/arshzahed/dev/launchpad/models/run2 \
--sequence_example_file=/Users/arshzahed/dev/launchpad/datasets/datasets/test/phase_try_1/training_melodies.tfrecord \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--num_training_steps=100

melody_rnn_generate \
--config=phase_rnn \
--run_dir=/Users/arshzahed/dev/launchpad/models/run2 \
--output_dir=/Users/arshzahed/dev/launchpad/generated/4 \
--num_outputs=5 \
--num_steps=256 \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--primer_melody="[60]"

melody_rnn_generate \
--config=phase_rnn \
--run_dir=/Users/arshzahed/dev/launchpad/models/run1 \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--bundle_file=/Users/arshzahed/dev/launchpad/models/phase_100.mag \
--save_generator_bundle

melody_rnn_generate \
--config=phase_rnn \
--bundle_file=/Users/arshzahed/dev/launchpad/models/phase_100.mag \
--output_dir=/tmp/melody_rnn/generated/3 \
--num_outputs=10 \
--num_steps=256 \
--primer_melody="[60]"

performance_rnn_create_dataset \
--config=phase_performance \
--input=/Users/arshzahed/dev/launchpad/datasets/datasets/phase1.tfrecord \
--output_dir=/Users/arshzahed/dev/launchpad/datasets/phpe_try1 \
--eval_ratio=0.10

