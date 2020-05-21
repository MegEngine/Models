class ShufflenetConfig:
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0625
    MOMENTUM = 0.9
    WEIGHT_DECAY = lambda n, p: \
        4e-5 if n.find("weight") >= 0 and len(p.shape) > 1 else 0
    EPOCHS = 240

    SCHEDULER = "Linear"


class ResnetConfig:
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0125
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    EPOCHS = 90

    SCHEDULER = "Multistep"
    SCHEDULER_STEPS = [30, 60, 80]
    SCHEDULER_GAMMA = 0.1


def get_config(arch: str):
    if "resne" in arch:  # both resnet and resnext
        return ResnetConfig()
    elif "shufflenet" in arch or "mobilenet" in arch:
        return ShufflenetConfig()
    else:
        raise ValueError("config for {} not exists".format(arch))


class ShufflenetFinetuneConfig:
    BATCH_SIZE = 128
    LEARNING_RATE = 0.000625
    MOMENTUM = 0.9
    WEIGHT_DECAY = lambda n, p: \
        4e-5 if n.find("weight") >= 0 and len(p.shape) > 1 else 0
    EPOCHS = 24

    SCHEDULER = "Linear"


class ResnetFinetuneConfig:
    BATCH_SIZE = 32
    LEARNING_RATE = 0.000125
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    EPOCHS = 9

    SCHEDULER = "Multistep"
    SCHEDULER_STEPS = [6,]
    SCHEDULER_GAMMA = 0.1


def get_finetune_config(arch: str):
    if "resne" in arch:  # both resnet and resnext
        return ResnetFinetuneConfig()
    elif "shufflenet" in arch or "mobilenet" in arch:
        return ShufflenetFinetuneConfig()
    else:
        raise ValueError("config for {} not exists".format(arch))
