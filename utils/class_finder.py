import argparse
import optimizers

def optimizer_class(class_name):
    if class_name not in optimizers.__all__:
       raise argparse.ArgumentTypeError("Invalid optimizer {}; choices: {}".format(
           class_name, optimizers.__all__))
    return getattr(optimizers, class_name)
