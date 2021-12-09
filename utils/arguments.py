import argparse


def get_parse_train():
  parser = argparse.ArgumentParser(
      description='Training'
  )
  parser.add_argument(
      'config_file',
      help='Path to the file that contains the training configuration'
  )
  parser.add_argument(
      'output_directory',
      help='Save output files in the given directory'
  )
  parser.add_argument(
      '--weight_file',
      default=None,
      help='Path to a previously trained model to continue training'
  )
  parser.add_argument(
      '--continue_from_epoch',
      default=0,
      type=int,
      help='Continue training from epoch'
  )
