import argparse


def parseArgument(mode = 'train'):
    if mode == 'train':
        parser = argparse.ArgumentParser(description = "Process training arguments for train_attntn.py")
        parser.add_argument("-model", type=str, required=True, help="specify the encoder model")
        parser.add_argument("-name", type=str, required=True, help="specify the name of this training")
        parser.add_argument("-resume_e", type=int, default=-1, help = "specify the epoch to resume at")
    else:
        parser = argparse.ArgumentParser(description = "Process training arguments for test.py")
        parser.add_argument("-model", required=True, help="specify the encoder model")
        parser.add_argument("-name", required=True, help="specify the name of the model")

    args = parser.parse_args()
    return args
