import argparse
import sys
import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants

print('Python Version {}'.format(sys.version))
print('Tensorflow Version {}'.format(tf.__version__))

parser = argparse.ArgumentParser()

parser.add_argument('--debug', type=bool, default=True, help='Wait for debugge attach')

parser.add_argument('--saved_model', type=str, default='./saved_model/1587139169',
                    help='Base directory for the model.')

parser.add_argument('--out', type=str, 
                    default='frozen_model.pb',
                    #default='C:\\data\\datasets\\imdb',
                    help='Path to the directory containing the imdb data tf record.')


def freeze_model(saved_model_dir, output_filename, output_node_names):
    output_graph_filename = os.path.join(saved_model_dir, output_filename)
    initializer_nodes = ''
    output_node_names = output_node_names
    freeze_graph.freeze_graph(
        input_saved_model_dir=saved_model_dir,
        output_graph=output_graph_filename,
        saved_model_tags = tag_constants.SERVING,
        output_node_names=output_node_names,
        initializer_nodes=initializer_nodes,
        input_graph=None,
        input_saver=False,
        input_binary=False,
        input_checkpoint=None,
        restore_op_name=None,
        filename_tensor_name=None,
        clear_devices=False,
        input_meta_graph=False,
    )
    print('graph frozen!')

def main(unused_argv):

    freeze_model(FLAGS.saved_model, FLAGS.out, output_node_names='pred_age,pred_gender')

if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.debug:
        # https://code.visualstudio.com/docs/python/debugging#_remote-debugging
        # Launch applicaiton on remote computer: 
        # > python3 -m ptvsd --host 0.0.0.0 --port 3000 --wait predict_imdb.py
        import ptvsd
        # Allow other computers to attach to ptvsd at this IP address and port.
        ptvsd.enable_attach(address=('0.0.0.0', 3000), redirect_output=True)
        # Pause the program until a remote debugger is attached
        print("Wait for debugger attach")
        ptvsd.wait_for_attach()
        print("Debugger Attached")

 